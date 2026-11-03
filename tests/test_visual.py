import numpy as np
import pytest
import torch

from worldpolicy.visual_data import (
    PROPRIO_INDICES,
    collect_visual_episode,
    proprio_from_state,
    visual_chunks,
)
from worldpolicy.visual_experiment import train_visual
from worldpolicy.visual_policy import VisualPolicy


def test_dual_camera_sync_and_information_boundary():
    episode = collect_visual_episode(12)
    episode.validate()
    assert episode.images.shape[1:] == (2, 64, 64, 3)
    assert len(episode.images) == len(episode.actions) == len(episode.timestamps)
    assert np.mean(np.abs(episode.images[0, 0].astype(float) - episode.images[0, 1])) > 1
    assert all(index not in PROPRIO_INDICES for index in (3, 4, 5, 25, 26, 27))
    state = np.arange(28, dtype=np.float32)
    changed = state.copy()
    changed[[3, 4, 5, 25, 26, 27]] = 1e6
    np.testing.assert_equal(proprio_from_state(state), proprio_from_state(changed))


def test_visual_policy_forward_backward_and_no_proprio_leakage():
    torch.set_num_threads(1)
    pixels = torch.randint(0, 255, (2, 2, 2, 3, 64, 64), dtype=torch.uint8)
    proprio = torch.randn(2, 2, 10)
    actions = torch.randn(2, 4, 4).clamp(-1, 1)
    vision_only = VisualPolicy("act", use_proprio=False)
    vision_only.eval()
    with torch.no_grad():
        a = vision_only.act(pixels, proprio)
        b = vision_only.act(pixels, proprio + 100)
    torch.testing.assert_close(a, b)
    multimodal = VisualPolicy("diffusion", use_proprio=True)
    loss = multimodal.loss(pixels, proprio, actions)
    loss.backward()
    assert torch.isfinite(loss) and multimodal.image_encoder[0].weight.grad is not None
    with pytest.raises(ValueError, match="shape"):
        multimodal.loss(pixels, torch.zeros(2, 3, 10), actions)


def test_visual_chunk_alignment():
    episode = collect_visual_episode(3)
    images, proprio, actions = visual_chunks([episode], history=2, horizon=4)
    assert images.shape[1:] == (2, 2, 64, 64, 3)
    assert proprio.shape[1:] == (2, 10)
    np.testing.assert_equal(images[0, 0], episode.images[0])
    np.testing.assert_equal(actions[0], episode.actions[:4])


def test_task_visible_cameras_show_target_and_object():
    from worldpolicy.fetch_push import FetchPush

    env = FetchPush(90000, camera_setup="task_visible")
    env.reset()
    image = env.camera_frames()
    env.close()
    assert image.shape == (2, 64, 64, 3)
    # The explicit red task marker and dark object are visible in both views.
    assert all(np.count_nonzero(frame[:, :, 0] > frame[:, :, 1] + 80) > 3 for frame in image)
    assert all(np.count_nonzero(np.mean(frame, axis=-1) < 80) > 3 for frame in image)


def test_object_shift_changes_only_reset_object_position():
    from worldpolicy.fetch_push import FetchPush

    base = FetchPush(90000, camera_setup="task_visible")
    shifted = FetchPush(90000, camera_setup="task_visible", object_shift=(.03, 0.))
    a, b = base.reset(), shifted.reset()
    np.testing.assert_allclose(b[3] - a[3], .03, atol=1e-5)
    np.testing.assert_allclose(a[25:28], b[25:28], atol=1e-6)
    base.close()
    shifted.close()


def test_visual_training_resume_restores_rng_and_optimizer(tmp_path):
    images = np.zeros((4, 2, 2, 64, 64, 3), dtype=np.uint8)
    proprio = np.zeros((4, 2, 10), dtype=np.float32)
    actions = np.zeros((4, 4, 4), dtype=np.float32)
    original = VisualPolicy("act", use_proprio=False)
    train_visual(original, images, proprio, actions, tmp_path, 101, 2, 5e-4, 31)
    resumed = VisualPolicy("act", use_proprio=False)
    train_visual(resumed, images, proprio, actions, tmp_path, 101, 2, 5e-4, 31, resume=True)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
