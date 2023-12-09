# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Functions for computing inverse kinematics on MuJoCo models."""

import collections
import copy

import mujoco
import numpy as np
from absl import logging

_INVALID_JOINT_NAMES_TYPE = (
    '`joint_names` must be either None, a list, a tuple, or a numpy array; '
    'got {}.')
_REQUIRE_TARGET_POS_OR_QUAT = (
    'At least one of `target_pos` or `target_quat` must be specified.')

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


def qpos_from_site_pose(model, data,
                        site_name,
                        target_pos=None,
                        target_quat=None,
                        joint_names=None,
                        tol=1e-14,
                        rot_weight=1.0,
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        max_steps=100):
    """Find joint positions that satisfy a target site position and/or rotation.
    # 找到满足目标位置和旋转的关节位置
  Args:
    model: mujoco model
    data: mujoco data
    site_name: A string specifying the name of the target site.
    # 一个3行的数组 指定期望笛卡尔位置
    target_pos: A (3,) numpy array specifying the desired Cartesian position of
      the site, or None if the position should be unconstrained (default).
      One or both of `target_pos` or `target_quat` must be specified.
    # 一个4行的数组 指定期望笛卡尔方向
    target_quat: A (4,) numpy array specifying the desired orientation of the
      site as a quaternion, or None if the orientation should be unconstrained
      (default). One or both of `target_pos` or `target_quat` must be specified.
    joint_names: (optional) A list, tuple or numpy array specifying the names of
      one or more joints that can be manipulated in order to achieve the target
      site pose. If None (default), all joints may be manipulated.
    tol: (optional) Precision goal for `qpos` (the maximum value of `err_norm`
      in the stopping criterion).
    # 确定旋转误差相对于平移误差的权重
    rot_weight: (optional) Determines the weight given to rotational error
      relative to translational error.
    regularization_threshold: (optional) L2 regularization will be used when
      inverting the Jacobian whilst `err_norm` is greater than this value.
    # 关节运动的二次惩罚系数
    regularization_strength: (optional) Coefficient of the quadratic penalty
      on joint movements.
    max_update_norm: (optional) The maximum L2 norm of the update applied to
      the joint positions on each iteration. The update vector will be scaled
      such that its magnitude never exceeds this value.
    progress_thresh: (optional) If `err_norm` divided by the magnitude of the
      joint position update is greater than this value then the optimization
      will terminate prematurely. This is a useful heuristic to avoid getting
      stuck in local minima.
    max_steps: (optional) The maximum number of iterations to perform.

  Returns:
  # 返回IKResult的命名元组
    An `IKResult` namedtuple with the following fields:
      # 关节位置
      qpos: An (nq,) numpy array of joint positions.
      err_norm: A float, the weighted sum of L2 norms for the residual
        translational and rotational errors.
      steps: An int, the number of iterations that were performed.
      success: Boolean, True if we converged on a solution within `max_steps`,
        False otherwise.

  Raises:
    ValueError: If both `target_pos` and `target_quat` are None, or if
      `joint_names` has an invalid type.
  """

    # 初始化 
    dtype = data.qpos.dtype
    # 此处将jac_pos 和 jac_rot 初始化
    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

    update_nv = np.zeros(model.nv, dtype=dtype)
    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    # Ensure that the Cartesian position of the site is up-to-date.
    mujoco.mj_fwdPosition(model, data)

    # Convert site name to index.
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
    # update them in place, so we can avoid indexing overhead in the main loop.
    # site的笛卡尔位置和旋转矩阵
    site_xpos = data.site_xpos[site_id]
    site_xmat = data.site_xmat[site_id]

    # This is an index into the rows of `update` and the columns of `jac`
    # that selects DOFs associated with joints that we are allowed to manipulate.
    if joint_names is None:
        dof_indices = slice(None)  # Update all DOFs.
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        # Find the indices of the DOFs belonging to each named joint. Note that
        # these are not necessarily the same as the joint IDs, since a single joint
        # may have >1 DOF (e.g. ball joints).
        indexer = model.dof_jntid.axes.row
        # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
        # indexer to map each joint name to the indices of its corresponding DOFs.
        dof_indices = indexer.convert_key_item(joint_names)
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    steps = 0
    success = False

    # 迭代搜索关节角度
    for steps in range(max_steps):

        err_norm = 0.0

        # 1. 计算误差
        if target_pos is not None:
            # Translational error.
            err_pos[:] = target_pos - site_xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # Rotational error.
            mujoco.mju_mat2Quat(site_xquat, site_xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            logging.debug('Converged after %i steps: err_norm=%3g', steps, err_norm)
            success = True
            break
        else:
            # 2. 计算雅可比矩阵
            mujoco.mj_jacSite(
                model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]

            reg_strength = (
                regularization_strength if err_norm > regularization_threshold
                else 0.0)

            # 3. 求解关节更新量delta q
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength=reg_strength)

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                logging.debug('Step %2i: err_norm / update_norm (%3g) > '
                              'tolerance (%3g). Halting due to insufficient progress',
                              steps, progress_criterion, progress_thresh)
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # Write the entries for the specified joints into the full `update_nv`
            # vector.
            update_nv[dof_indices] = update_joints

            # Update `physics.qpos`, taking quaternions into account.
            # 4. 更新关节角度
            mujoco.mj_integratePos(model, data.qpos, update_nv, 1)
            # mujoco.mj_integratePos(model, qpos_copy, update_nv, 1)

            # Compute the new Cartesian position of the site.
            mujoco.mj_fwdPosition(model, data)

            logging.debug('Step %2i: err_norm=%-10.3g update_norm=%-10.3g',
                          steps, err_norm, update_norm)

    if not success and steps == max_steps - 1:
        logging.warning('Failed to converge after %i steps: err_norm=%3g',
                        steps, err_norm)

    qpos = copy.deepcopy(data.qpos)

    return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.

  Args:
    jac_joints: The Jacobian of the end effector with respect to the joints. A
      numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
      and `nv` is the number of degrees of freedom.
    delta: The desired end-effector delta. A numpy array of shape `(3,)` or
      `(6,)` containing either position deltas, rotation deltas, or both.
    regularization_strength: (optional) Coefficient of the quadratic penalty
      on joint movements. Default is zero, i.e. no regularization.

  Returns:
    An `(nv,)` numpy array of joint velocities.

  Reference:
    Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
    transpose, pseudoinverse and damped least squares methods.
    https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
  """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
