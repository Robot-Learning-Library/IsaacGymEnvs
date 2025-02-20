# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for insertion task.

Inherits insertion environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskInsertion

Only the environment is provided; training a successful RL policy is an open research problem left to the user.
"""

import hydra
import math
import omegaconf
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_insertion import FactoryEnvInsertion
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
from isaacgymenvs.utils import torch_jit_utils


class FactoryTaskInsertion(FactoryEnvInsertion, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize task superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer != None:
            self._set_viewer_params()
        if self.cfg_base.mode.export_scene:
            self.export_scene(label='franka_task_insertion')

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskInsertionPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        # target_heights = self.cfg_base.env.table_height + self.bolt_head_heights + self.nut_heights * 0.5
        # self.target_pos = target_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        
        # Grasp pose tensors
        self.plug_heights = 0.005  # this should be given from yaml file
        self.socket_heights = 0.0
        plug_grasp_heights = self.socket_heights + self.plug_heights * 0.5  # nut COM
        self.plug_grasp_pos_local = plug_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1))
        self.plug_grasp_quat_local = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(
            self.num_envs, 1)  # this is right when plug does not fall; after falling the best grasping orientation will not be this one

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_gripper = torch.zeros((self.num_envs, self.cfg_task.rl.num_keypoints, 3),
                                             dtype=torch.float32,
                                             device=self.device)
        self.keypoints_plug = torch.zeros_like(self.keypoints_gripper, device=self.device)

        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
               

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        # Compute pose of plug grasping frame
        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_quat,
                                                                             self.plug_pos,
                                                                             self.plug_grasp_quat_local,
                                                                             self.plug_grasp_pos_local)

        # Compute pos of keypoints on gripper and nut in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_midpoint_quat,
                                                                        self.fingertip_midpoint_pos,
                                                                        self.identity_quat,
                                                                        keypoint_offset.repeat(self.num_envs, 1))[1]
            self.keypoints_plug[:, idx] = torch_jit_utils.tf_combine(self.plug_grasp_quat,
                                                                    self.plug_grasp_pos,
                                                                    self.identity_quat,
                                                                    keypoint_offset.repeat(self.num_envs, 1))[1]


        self.fingerpad_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_franka_table.franka_finger_length - self.asset_info_franka_table.franka_fingerpad_length * 0.5,
                                                                 device=self.device)
        # self.finger_plug_keypoint_dist = self._get_keypoint_dist(body='finger_plug')
        # self.plug_keypoint_dist = self._get_keypoint_dist(body='plug')

        # self.plug_com_pos = fc.translate_along_local_z(pos=self.plug_pos,
        #                                               quat=self.plug_quat,
        #                                             #   offset=self.bolt_head_heights + self.nut_heights * 0.5,
        #                                               offset = 0.002,  # needs more accurate value here
        #                                               device=self.device)

        self.plug_dist_to_fingerpads = torch.norm(self.fingerpad_midpoint_pos - self.plug_pos, p=2, dim=-1)  # distance between plug and midpoint between centers of fingerpads
        self.finger_plug_keypoint_dist = self.plug_dist_to_fingerpads
        self.socket_dist_to_plug = torch.norm(self.plug_pos - self.socket_pos, p=2,
                                                 dim=-1)  # distance between socket and plug

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _get_keypoint_dist(self, body):
        """Get keypoint distance."""

        if body == 'finger_plug':
            keypoint_dist = torch.sum(torch.norm(self.keypoints_plug - self.keypoints_gripper, p=2, dim=-1), dim=-1)

        return keypoint_dist

    def _get_keypoint_dist_deprecated(self, body):
        """Get keypoint distances."""

        axis_length = self.asset_info_franka_table.franka_hand_length + self.asset_info_franka_table.franka_finger_length

        if body == 'finger' or body == 'plug':
            # Keypoint distance between finger/nut and target
            if body == 'finger':
                self.keypoint1 = self.fingertip_midpoint_pos
                self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                            quat=self.fingertip_midpoint_quat,
                                                            offset=-axis_length,
                                                            device=self.device)

            # elif body == 'plug':
            #     self.keypoint1 = self.plug_pos
            #     self.keypoint2 = fc.translate_along_local_z(pos=self.plug_pos,
            #                                                 quat=self.plug_quat,
            #                                                 offset=axis_length,
            #                                                 device=self.device)

            # self.keypoint1_targ = self.target_pos
            # self.keypoint2_targ = self.keypoint1_targ + torch.tensor([0.0, 0.0, axis_length], device=self.device)

        elif body == 'finger_plug':
            # Keypoint distance between finger and plug
            self.keypoint1 = self.fingerpad_midpoint_pos
            self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                        quat=self.fingertip_midpoint_quat,
                                                        offset=-axis_length,
                                                        device=self.device)

            self.keypoint1_targ = self.plug_pos
            self.keypoint2_targ = fc.translate_along_local_z(pos=self.plug_pos,
                                                             quat=self.plug_quat,
                                                             offset=axis_length,
                                                             device=self.device)

        self.keypoint3 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 1.0 / 3.0
        self.keypoint4 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 2.0 / 3.0
        self.keypoint3_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 1.0 / 3.0
        self.keypoint4_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 2.0 / 3.0
        keypoint_dist = torch.norm(self.keypoint1_targ - self.keypoint1, p=2, dim=-1) \
                        + torch.norm(self.keypoint2_targ - self.keypoint2, p=2, dim=-1) \
                        + torch.norm(self.keypoint3_targ - self.keypoint3, p=2, dim=-1) \
                        + torch.norm(self.keypoint4_targ - self.keypoint4, p=2, dim=-1)

        return keypoint_dist

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        # if control the gripper with action
        # self._apply_actions_as_ctrl_targets(actions=self.actions,
        #                                     ctrl_target_gripper_dof_pos=self.actions[:, -1],
        #                                     do_scale=True)

        less_than_half_step = (self.progress_buf[0] < int((self.max_episode_length - 1)/2))
        if less_than_half_step: # keep the gripper opening
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                                do_scale=True)
        else:  # keep the gripper close
            # self.actions = torch.zeros_like(self.actions).to(self.device)   # keep static
            self._apply_actions_as_ctrl_targets(actions=self.actions,
                                    ctrl_target_gripper_dof_pos=0.,
                                    do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        is_half_step = (self.progress_buf[0] == int((self.max_episode_length - 1)/2))
        if self.cfg_task.env.close_and_lift:
            # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
            if is_half_step:
                self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
                self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

        if is_last_step:  # open gripper and lift up to see if the plug is already in socket
                self._open_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
                self._lift_gripper(franka_gripper_width=self.asset_info_franka_table.franka_gripper_width_max, sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""
        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,  # 3 dim
                       self.fingertip_midpoint_quat, # 4 dim
                       self.fingertip_midpoint_linvel,  # 3 dim
                       self.fingertip_midpoint_angvel,  # 3 dim
                       self.plug_pos,   # 3 dim
                       self.plug_quat,  # 4 dim

                    #    self.plug_linvel,  # 3 dim
                    #    self.plug_angvel,  # 3 dim
                    #    self.plug_grasp_pos,  # 3 dim
                    #    self.plug_grasp_quat, # 4 dim

                       self.socket_pos,  # 3 dim
                       self.socket_quat, # 4 dim
                    #    self.peg_diameters,  # 1 dim
                    #    self.hole_diameters  # 1 dim
                       ]

        if self.cfg_task.rl.add_obs_finger_force:
            obs_tensors += [self.left_finger_force, self.right_finger_force]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)

        return self.obs_buf  # shape = (num_envs, num_observations)

    def _get_curr_successes(self):
        """Get success mask at current timestep."""

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If nut is close enough to target pos
        is_close = torch.where(self.socket_dist_to_plug < torch.ones_like(self.socket_dist_to_plug) * 0.005,
                               torch.ones_like(curr_successes),
                               torch.zeros_like(curr_successes))

        curr_successes = torch.logical_or(curr_successes, is_close)

        return curr_successes
        
    def _get_curr_failures(self, curr_successes):
        """Get failure mask at current timestep."""
        # TODO
        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        # If max episode length has been reached
        self.is_expired = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
                                      torch.ones_like(curr_failures),
                                      curr_failures)

        # If plug is too far from socket
        self.is_far = torch.where(self.socket_dist_to_plug > self.cfg_task.rl.far_error_thresh,
                                  torch.ones_like(curr_failures),
                                  curr_failures)

        # If plug has slipped (distance-based definition) # TODO plug height is not provided
        # self.is_slipped = \
        #     torch.where(
        #         self.plug_dist_to_fingerpads > self.asset_info_franka_table.franka_fingerpad_length * 0.5 + self.nut_heights.squeeze(-1) * 0.5,
        #         torch.ones_like(curr_failures),
        #         curr_failures)
        # self.is_slipped = torch.logical_and(self.is_slipped, torch.logical_not(curr_successes))  # ignore slip if successful

        # If nut has fallen (i.e., if nut XY pos has drifted from center of bolt and nut Z pos has drifted below top of bolt)
        # self.is_fallen = torch.logical_and(
        #     torch.norm(self.nut_com_pos[:, 0:2], p=2, dim=-1) > self.bolt_widths.squeeze(-1) * 0.5,
        #     self.nut_com_pos[:, 2] < self.cfg_base.env.table_height + self.bolt_head_heights.squeeze(
        #         -1) + self.bolt_shank_lengths.squeeze(-1) + self.nut_heights.squeeze(-1) * 0.5)

        curr_failures = torch.logical_or(curr_failures, self.is_expired)
        # curr_failures = torch.logical_or(curr_failures, self.is_far)
        # curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        # curr_failures = torch.logical_or(curr_failures, self.is_fallen)

        return curr_failures

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""
        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_successes()
        curr_failures = self._get_curr_failures(curr_successes)

        self._update_rew_buf(curr_successes)
        self._update_reset_buf(curr_failures)

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""
        # In this policy, episode length is constant across all envs
        after_half_step = (self.progress_buf[0] == int((self.max_episode_length - 1)/2+1))        

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if self.progress_buf[0] < after_half_step:
            keypoint_reward = -self.finger_plug_keypoint_dist
        else:
            keypoint_reward = -self.socket_dist_to_plug

        action_penalty = torch.norm(self.actions, p=2, dim=-1)

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          + curr_successes * self.cfg_task.rl.success_bonus

        if after_half_step:        
            # Check if plug is picked up and above table
            lift_success = self._check_lift_success()
            self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            # self.extras['successes'] = torch.mean(lift_success.float())
            print('lift success rate: ', torch.mean(lift_success.float()) )

        if is_last_step:
            self.rew_buf[:] += curr_successes * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(curr_successes.float())
            print('final success rate: ', self.extras['successes'].item())

    def _update_reset_buf(self, curr_failures):
        """Assign environments for reset if successful or failed."""
        self.reset_buf[:] = curr_failures


    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka to middle of joint limits, plus joint noise
        franka_dof_props = self.gym.get_actor_dof_properties(self.env_ptrs[0],
                                                             self.franka_handles[0])  # same across all envs
        lower_lims = franka_dof_props['lower']
        upper_lims = franka_dof_props['upper']
        self.dof_pos[:, 0:self.franka_num_dofs] = torch.tensor((lower_lims + upper_lims) * 0.5, device=self.device) \
                                                  + (torch.rand((self.num_envs, 1),
                                                                device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.joint_noise * math.pi / 180

        self.dof_vel[env_ids, 0:self.franka_num_dofs] = 0.0

        franka_actor_ids_sim_int32 = self.franka_actor_ids_sim.to(dtype=torch.int32, device=self.device)[env_ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_actor_ids_sim_int32),
                                              len(franka_actor_ids_sim_int32))

        self.ctrl_target_dof_pos[env_ids, 0:self.franka_num_dofs] = self.dof_pos[env_ids, 0:self.franka_num_dofs]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ctrl_target_dof_pos))

    def _reset_object(self, env_ids):
        """Reset root state of plug."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        reset_num_envs = len(env_ids)
        if self.cfg_task.randomize.initial_state == 'random':
            self.root_pos[env_ids, self.plug_actor_id_env] = \
                torch.cat(((torch.rand((reset_num_envs, 1), device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.plug_noise_xy,
                           self.cfg_task.randomize.plug_bias_y + (torch.rand((reset_num_envs, 1), device=self.device) * 2.0 - 1.0) * self.cfg_task.randomize.plug_noise_xy,
                           torch.ones((reset_num_envs, 1), device=self.device) * (self.cfg_base.env.table_height + self.cfg_task.randomize.plug_bias_z)), dim=1)
            self.root_quat[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32,
                                                                        device=self.device).repeat(len(env_ids), 1)

            self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        elif self.cfg_task.randomize.initial_state == 'goal':
            self.root_pos[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height],
                                                                          device=self.device)

        self.root_linvel[env_ids, self.plug_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.plug_actor_id_env] = 0.0

        plug_actor_ids_sim_int32 = self.plug_actor_ids_sim.to(dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_actor_ids_sim_int32[env_ids]),
                                                     len(plug_actor_ids_sim_int32[env_ids]))

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        # if control the gripper with action as well
        # self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos.unsqueeze(-1).repeat(1,2) # gripper action is symmetric for two pads: (a,a)
        # if do_scale:
        #     self.ctrl_target_gripper_dof_pos = (self.ctrl_target_gripper_dof_pos + 1.) * 0.5  # range [0,1]
        
        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _check_lift_success(self, height_thresh=0.02):
        """Check if nut is above table by more than specified multiple times height of nut."""
        lift_success = torch.where(
            self.plug_pos[:, 2] > self.cfg_base.env.table_height + height_thresh,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _close_gripper(self, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self, franka_gripper_width=0.0, lift_distance=0.03, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
            self.render()
            self.gym.simulate(self.sim)