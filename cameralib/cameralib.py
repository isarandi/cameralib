import copy
import functools

import boxlib
import cv2
import numpy as np
import transforms3d


def point_transform(f):
    """Decorator to make a function, which transforms multiple points, also accept a single point,
    as well as lists, tuples etc. that can be converted by np.asarray."""

    @functools.wraps(f)
    def wrapped(self, points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2:
            return f(self, points, *args, **kwargs)

        reshaped = np.reshape(points, [-1, points.shape[-1]])
        reshaped_result = f(self, reshaped, *args, **kwargs)
        return np.reshape(reshaped_result, [*points.shape[:-1], reshaped_result.shape[-1]])

    return wrapped


def camera_transform(f):
    """Decorator to make a function, which transforms the camera,
    also accept an 'inplace' argument."""

    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        inplace = kwargs.pop('inplace', True)
        if inplace:
            return f(self, *args, **kwargs)
        else:
            camcopy = self.copy()
            f(camcopy, *args, **kwargs)
            return camcopy

    return wrapped


class Camera:
    def __init__(
            self, optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
            distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None,
            trans_after_rot=None, dtype=np.float32):
        """Pinhole camera with extrinsic and intrinsic calibration with optional distortions.

        The camera coordinate system has the following axes:
          x points to the right
          y points down
          z points forwards

        The world z direction is assumed to point up by default, but `world_up` can also be
         specified differently.

        Args:
            optical_center: position of the camera in world coordinates (eye point)
            rot_world_to_cam: 3x3 rotation matrix for transforming column vectors
                from being expressed in world reference frame to being expressed in camera
                reference frame as follows:
                column_point_cam = rot_matrix_world_to_cam @ (column_point_world - optical_center)
            intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
                coordinates in image (pixel) space. Its last row must be (0,0,1).
            distortion_coeffs: parameters describing radial and tangential lens distortions,
                following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
                if the camera has no distortion.
            world_up: a world vector that is designated as "pointing up".
            extrinsic_matrix: 4x4 extrinsic transformation matrix as an alternative to
                providing `optical_center` and `rot_world_to_cam`.
            trans_after_rot: translation vector to apply after the rotation
                (alternative to optical_center, which is a negative translation before the rotation)
        """

        if optical_center is not None and extrinsic_matrix is not None:
            raise Exception(
                'Provide only one of `optical_center`, `trans_after_rot` or `extrinsic_matrix`!')
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise Exception('Provide only one of `rot_world_to_cam` or `extrinsic_matrix`!')

        if (optical_center is None) and (trans_after_rot is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3, dtype=dtype)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3, dtype=dtype)

        if extrinsic_matrix is not None:
            self.R = np.asarray(extrinsic_matrix[:3, :3], dtype=dtype)
            self.t = -self.R.T @ extrinsic_matrix[:3, 3].astype(dtype)
        else:
            self.R = np.asarray(rot_world_to_cam, dtype=dtype)
            if optical_center is not None:
                self.t = np.asarray(optical_center, dtype=dtype)
            else:
                self.t = -self.R.T @ np.asarray(trans_after_rot, dtype=dtype)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=dtype)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=dtype)

        self.world_up = np.asarray(world_up, dtype=dtype)

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise Exception(f'Bottom row of intrinsic matrix must be (0,0,1), '
                            f'got {self.intrinsic_matrix[2, :]}.')

    # Methods to transform between coordinate systems (world, camera, image)
    @point_transform
    def camera_to_image(self, points):
        """Transform points from 3D camera coordinate space to image space.
        The steps involved are:
            1. Projection
            2. Distortion (radial and tangential)
            3. Applying focal length and principal point (intrinsic matrix)

        Equivalently:

        projected = points[:, :2] / points[:, 2:]

        if self.distortion_coeffs is not None:
            r2 = np.sum(projected[:, :2] ** 2, axis=1, keepdims=True)

            k = self.distortion_coeffs[[0, 1, 4]]
            radial = 1 + np.hstack([r2, r2 ** 2, r2 ** 3]) @ k

            p_flipped = self.distortion_coeffs[[3, 2]]
            tagential = projected @ (p_flipped * 2)
            distorted = projected * np.expand_dims(radial + tagential, -1) + p_flipped * r2
        else:
            distorted = projected

        return distorted @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]
        """

        if self.distortion_coeffs is not None:
            return project_points(points, self.distortion_coeffs, self.intrinsic_matrix)
        else:
            projected = points[..., :2] / points[..., 2:]
            return projected @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]

    @point_transform
    def world_to_camera(self, points):
        return (points - self.t) @ self.R.T

    @point_transform
    def camera_to_world(self, points):
        return points @ self.R + self.t

    @point_transform
    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    @point_transform
    def image_to_camera(self, points, depth=1):
        if self.distortion_coeffs is None:
            normalized_points = (
                ((points - self.intrinsic_matrix[:2, 2]) @
                 np.linalg.inv(self.intrinsic_matrix[:2, :2])))
            return cv2.convertPointsToHomogeneous(normalized_points)[:, 0, :] * depth

        points = np.expand_dims(np.asarray(points, np.float32), 0)
        new_image_points = cv2.undistortPoints(
            points, self.intrinsic_matrix, self.distortion_coeffs, None, None, None)
        return cv2.convertPointsToHomogeneous(new_image_points)[:, 0, :] * depth

    @point_transform
    def image_to_world(self, points, camera_depth=1):
        return self.camera_to_world(self.image_to_camera(points, camera_depth))

    @point_transform
    def is_visible(self, world_points, imsize):
        imsize = np.asarray(imsize)
        cam_points = self.world_to_camera(world_points)
        im_points = self.camera_to_image(cam_points)

        is_within_frame = np.all(np.logical_and(0 <= im_points, im_points < imsize), axis=1)
        is_in_front_of_camera = cam_points[..., 2] > 0
        return np.logical_and(is_within_frame, is_in_front_of_camera)

    # Methods to transform the camera parameters
    @camera_transform
    def shift_image(self, offset):
        """Adjust intrinsics so that the projected image is shifted by `offset`.

        Args:
            offset: an (x, y) offset vector. Positive values mean that the resulting image will
                shift towards the left and down.
        """
        self.intrinsic_matrix[:2, 2] += offset

    @camera_transform
    def shift_to_desired(self, current_coords_of_the_point, target_coords_of_the_point):
        """Shift the principal point such that what's currently at `desired_center_image_point`
        will be shown at `target_coords_of_the_point`.

        Args:
            current_coords_of_the_point: current location of the point of interest in the image
            target_coords_of_the_point: desired location of the point of interest in the image
        """

        self.intrinsic_matrix[:2, 2] += (target_coords_of_the_point - current_coords_of_the_point)

    @camera_transform
    def reset_roll(self):
        """Roll the camera upright by turning along the optical axis to align the vertical image
        axis with the vertical world axis (world up vector), as much as possible.
        """

        self.R[:, 0] = unit_vec(np.cross(self.R[:, 2], self.world_up))
        self.R[:, 1] = np.cross(self.R[:, 0], self.R[:, 2])

    @camera_transform
    def orbit_around(self, world_point_pivot, angle_radians, axis='vertical'):
        """Rotate the camera around a vertical or horizontal axis passing through `world point` by
        `angle_radians`.

        Args:
            world_point_pivot: the world coordinates of the pivot point to turn around
            angle_radians: the amount to rotate
            axis: 'vertical' or 'horizontal'.
        """

        if axis == 'vertical':
            axis = self.world_up
        else:
            lookdir = self.R[2]
            axis = unit_vec(np.cross(lookdir, self.world_up))

        rot_matrix = cv2.Rodrigues(axis * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = (rot_matrix @ (self.t - world_point_pivot)) + world_point_pivot

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = self.R @ rot_matrix.T

    @camera_transform
    def rotate(self, yaw=0, pitch=0, roll=0):
        """Rotate this camera by yaw, pitch, roll Euler angles in radians,
        relative to the current camera frame."""
        camera_rotation = transforms3d.euler.euler2mat(yaw, pitch, roll, 'ryxz')

        # The coordinates rotate according to the inverse of how the camera itself rotates
        point_coordinate_rotation = camera_rotation.T
        self.R = point_coordinate_rotation @ self.R

    def get_pitch_roll(self):

        yaw, pitch, roll = transforms3d.euler.mat2euler(self.R, 'ryxz')
        return pitch, roll

    @camera_transform
    def zoom(self, factor):
        """Zooms the camera (factor > 1 makes objects look larger),
        while keeping the principal point fixed (scaling anchor is the principal point)."""
        self.intrinsic_matrix[:2, :2] *= np.expand_dims(np.float32(factor), -1)

    @camera_transform
    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(np.float32(factor), -1)

    @camera_transform
    def undistort(self):
        self.distortion_coeffs = None

    @camera_transform
    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]], np.float32)
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    @camera_transform
    def horizontal_flip(self):
        self.R[0] *= -1

    @camera_transform
    def horizontal_flip_image(self, imshape):
        self.horizontal_flip()
        self.intrinsic_matrix[0, 2] = imshape[1] - self.intrinsic_matrix[0, 2]

    @camera_transform
    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = np.float32([imshape[1] / 2, imshape[0] / 2])

    @camera_transform
    def shift_to_center(self, desired_center_image_point, imshape):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        current_coords_of_the_point = desired_center_image_point
        target_coords_of_the_point = np.float32([imshape[1], imshape[0]]) / 2
        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    @camera_transform
    def turn_towards(self, target_image_point=None, target_world_point=None):
        """Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll)."""

        assert (target_image_point is None) != (target_world_point is None)
        if target_image_point is not None:
            target_world_point = self.image_to_world(target_image_point)

        new_z = unit_vec(target_world_point - self.t)
        new_x = unit_vec(np.cross(new_z, self.world_up))
        new_y = np.cross(new_z, new_x)

        # row_stack because we need the inverse transform (we make a matrix that transforms
        # points from one coord system to another), which is the same as the transpose
        # for rotation matrices.
        self.R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)

    # Getters
    def get_projection_matrix(self):
        extrinsic_projection = np.append(self.R, -self.R @ np.expand_dims(self.t, 1), axis=1)
        return self.intrinsic_matrix @ extrinsic_projection

    def get_extrinsic_matrix(self):
        return np.block(
            [[self.R, -self.R @ np.expand_dims(self.t, -1)], [0, 0, 0, 1]]).astype(np.float32)

    def get_fov(self, imshape):
        focals = np.diagonal(self.intrinsic_matrix)[:2]
        return np.rad2deg(2 * np.arctan(np.max(imshape[:2] / (2 * focals))))

    def get_distortion_coeffs(self):
        if self.distortion_coeffs is None:
            return np.zeros(shape=(5,), dtype=np.float32)
        return self.distortion_coeffs

    def allclose(self, other_camera):
        """Check if all parameters of this camera are close to corresponding parameters
        of `other_camera`.

        Args:
            other_camera: the camera to compare to.

        Returns:
            True if all parameters are close, False otherwise.
        """
        return (np.allclose(self.intrinsic_matrix, other_camera.intrinsic_matrix) and
                np.allclose(self.R, other_camera.R) and np.allclose(self.t, other_camera.t) and
                allclose_or_nones(self.distortion_coeffs, other_camera.distortion_coeffs))

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_fov(fov_degrees, imshape, world_up=(0, -1, 0)):
        intrinsics = intrinsics_from_fov(fov_degrees, imshape)
        return Camera(intrinsic_matrix=intrinsics, world_up=world_up)

    @staticmethod
    def create2D(imshape=(0, 0)):
        """Create a camera for expressing 2D transformations by using intrinsics only.

        Args:
            imshape: height and width, the principal point of the intrinsics is set at the middle
                of this image size.

        Returns:
            The new camera.
        """

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]
        return Camera(intrinsic_matrix=intrinsics)

    @property
    def optical_center(self):
        return self.t


def intrinsics_from_fov(fov_degrees, imshape):
    f = np.max(imshape[:2]) / (np.tan(np.deg2rad(fov_degrees) / 2) * 2)
    intrinsics = np.array(
        [[f, 0, imshape[1] / 2],
         [0, f, imshape[0] / 2],
         [0, 0, 1]], np.float32)
    return intrinsics


def reproject_image_points(points, old_camera, new_camera):
    """Transforms keypoints of an image captured with `old_camera` to the corresponding
    keypoints of an image captured with `new_camera`.
    The world position (optical center) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output image."""

    if (old_camera.distortion_coeffs is None and new_camera.distortion_coeffs is None and
            points.ndim == 2):
        return reproject_image_points_fast(points, old_camera, new_camera)

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics = (
                new_camera.intrinsic_matrix @ np.linalg.inv(old_camera.intrinsic_matrix))
        return points @ relative_intrinsics[:2, :2].T + relative_intrinsics[:2, 2]

    world_points = old_camera.image_to_world(points)
    return new_camera.world_to_image(world_points)


def reproject_image(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, antialias_factor=1, dst=None):
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Ignores the issue of aliasing altogether.

    Args:
        image: the input image
        old_camera: the camera that captured `image`
        new_camera: the camera that should capture the newly returned image
        output_imshape: (height, width) for the output image
        border_mode: OpenCV border mode for treating pixels outside `image`
        border_value: OpenCV border value for treating pixels outside `image`
        interp: OpenCV interpolation to be used for resampling.
        antialias_factor: If larger than 1, first render a higher resolution output image
            that is `antialias_factor` times larger than `output_imshape` and subsequently resize
            it by 'area' interpolation to the desired size.
        dst: destination array (optional)

    Returns:
        The new image.
    """
    if antialias_factor == 1:
        return reproject_image_aliased(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp,
            dst=dst)

    new_camera = new_camera.copy()
    a = antialias_factor
    new_camera.scale_output(a)
    new_camera.intrinsic_matrix[:2, 2] += (a - 1) / 2
    intermediate_imshape = (a * output_imshape[0], a * output_imshape[1])
    result = reproject_image_aliased(
        image, old_camera, new_camera, intermediate_imshape, border_mode, border_value, interp)
    return cv2.resize(
        result, dsize=(output_imshape[1], output_imshape[0]),
        interpolation=cv2.INTER_AREA, dst=dst)


def reproject_image_aliased(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, dst=None):
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Aliasing issues are ignored.
    """

    if interp is None:
        interp = cv2.INTER_LINEAR

    if old_camera.distortion_coeffs is None and new_camera.distortion_coeffs is None:
        return reproject_image_fast(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp, dst)

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    output_size = (output_imshape[1], output_imshape[0])

    # 1. Simplest case: if only the intrinsics have changed we can use an affine warp
    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics_inv = np.linalg.solve(
            new_camera.intrinsic_matrix.T, old_camera.intrinsic_matrix.T).T
        return cv2.warpAffine(
            image, relative_intrinsics_inv[:2], output_size, flags=cv2.WARP_INVERSE_MAP | interp,
            borderMode=border_mode, borderValue=border_value, dst=dst)

    # 2. The general case handled by transforming the coordinates of every pixel
    # (i.e. computing the source pixel coordinates for each destination pixel)
    # and remapping (i.e. resampling the image at the resulting coordinates)
    new_maps = get_grid_coords((output_imshape[0], output_imshape[1]))
    newim_coords = new_maps.reshape([-1, 2])

    if new_camera.distortion_coeffs is None:
        partial_homography = (
                old_camera.R @ np.linalg.inv(new_camera.R) @
                np.linalg.inv(new_camera.intrinsic_matrix))
        new_im_homogeneous = np.squeeze(cv2.convertPointsToHomogeneous(newim_coords), axis=1)
        old_camera_coords = new_im_homogeneous @ partial_homography.T
        oldim_coords = old_camera.camera_to_image(old_camera_coords)
    else:
        world_coords = new_camera.image_to_world(newim_coords)
        oldim_coords = old_camera.world_to_image(world_coords)

    old_maps = oldim_coords.reshape(new_maps.shape).astype(np.float32)
    map1 = old_maps[..., 0]
    map2 = old_maps[..., 1]

    remapped = cv2.remap(
        image, map1, map2, interp, borderMode=border_mode, borderValue=border_value, dst=dst)

    if remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)
    return remapped


def reproject_mask(
        mask, old_camera, new_camera, dst_shape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, antialias_factor=1, dst=None):
    # binarize to 0 vs 255 (assumed to be binary already, but perhaps 0 vs 1)
    mask = (mask != 0).astype(np.uint8) * 255
    new_mask = reproject_image(
        mask, old_camera, new_camera, dst_shape, border_mode, border_value, interp,
        antialias_factor, dst)
    new_mask //= 128
    return new_mask


def allclose_or_nones(a, b):
    """Check if all corresponding values in arrays a and b are close to each other in the sense of
    np.allclose, or both a and b are None, or one is None and the other is filled with zeros.
    """

    if a is None and b is None:
        return True

    if a is None:
        return cv2.countNonZero(b) == 0

    if b is None:
        return cv2.countNonZero(a) == 0

    return np.allclose(a, b)


# Optional to make it faster perhaps
# @numba.njit()
def project_points_simple(points, dist_coeff, intrinsic_matrix):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    points = points.astype(np.float32)
    proj = points[..., :2] / points[..., 2:]
    r2 = np.sum(proj * proj, axis=1)
    distorter = (
            ((dist_coeff[4] * r2 + dist_coeff[1]) * r2 + dist_coeff[0]) * r2 +
            np.float32(1.0) + np.sum(proj * (np.float32(2.0) * dist_coeff[3:1:-1]), axis=1))
    proj[:] = (
            proj * np.expand_dims(distorter, 1) + np.expand_dims(r2, 1) * dist_coeff[3:1:-1])
    return (proj @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)


def project_points(points, dist_coeff, intrinsic_matrix):
    # coefficient order: k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
    dist_coeff = np.pad(dist_coeff, (0, 12 - len(dist_coeff)))
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    points = points.astype(np.float32)
    proj = points[..., :2] / points[..., 2:]
    r2 = np.sum(proj * proj, axis=1)
    distorter = (
            ((((dist_coeff[4] * r2 + dist_coeff[1]) * r2 + dist_coeff[0]) * r2 + np.float32(1.0)) /
             (((dist_coeff[7] * r2 + dist_coeff[6]) * r2 + dist_coeff[5]) * r2 + np.float32(1.0))) +
            + np.float32(2.0) * np.sum(proj * dist_coeff[3:1:-1], axis=1))

    proj[:] = (
            proj * np.expand_dims(distorter, 1) +
            (dist_coeff[9:12:2] * np.expand_dims(r2, 1) +
             dist_coeff[8:11:2] + dist_coeff[3:1:-1]) * np.expand_dims(r2, 1))

    # Scheimpflug extension from opencv is not used
    # For that, one would need to implement tilt distortion
    # proj[:] = tilt_distort(proj, dist_coeff[12:14])
    return (proj @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)


@functools.lru_cache(5)
def get_grid_coords(output_imshape):
    """Return a meshgrid of coordinates for the image shape `output_imshape` (height, width).

    Returns
        Meshgrid of shape [height, width, 2], with the x and y coordinates (in this order)
            along the last dimension. DType float32.
    """
    y, x = np.mgrid[:output_imshape[0], :output_imshape[1]].astype(np.float32)
    return np.stack([x, y], axis=-1)


def reproject_image_fast(
        image, old_camera, new_camera, output_imshape, border_mode=None, border_value=None,
        interp=cv2.INTER_LINEAR, dst=None):
    """Like reproject_image, but assumes there are no lens distortions."""

    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = np.linalg.solve(new_matrix.T, old_matrix.T).T.astype(np.float32)

    if border_mode is None:
        border_mode = cv2.BORDER_CONSTANT
    if border_value is None:
        border_value = 0

    remapped = cv2.warpPerspective(
        image, homography, (output_imshape[1], output_imshape[0]),
        flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border_mode,
        borderValue=border_value, dst=dst)

    if remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)
    return remapped


def reproject_image_points_fast(points, old_camera, new_camera):
    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = np.linalg.solve(old_matrix.T, new_matrix.T).T.astype(np.float32)
    pointsT = homography[:, :2] @ points.T + homography[:, 2:]
    pointsT = pointsT[:2] / pointsT[2:]
    return pointsT.T


def reproject_box(old_box, old_camera, new_camera):
    return (reproject_box_corners(old_box, old_camera, new_camera) +
            reproject_box_side_midpoints(old_box, old_camera, new_camera)) / 2


def reproject_box_corners(old_box, old_camera, new_camera):
    old_corners = boxlib.corners(old_box)
    new_corners = reproject_image_points(old_corners, old_camera, new_camera)
    return boxlib.bb_of_points(new_corners)


def reproject_box_side_midpoints(old_box, old_camera, new_camera):
    old_side_midpoints = boxlib.side_midpoints(old_box)
    new_side_midpoints = reproject_image_points(old_side_midpoints, old_camera, new_camera)
    return boxlib.bb_of_points(new_side_midpoints)


def unit_vec(v):
    return v / np.linalg.norm(v)
