from types import SimpleNamespace

from PIL import Image

from trainer.fsdp2.bagel_probe_mixin import BagelProbeMixin


class _FixedResizeTransform:
    def __init__(self, target_size):
        self.target_size = tuple(target_size)

    def resize_transform(self, image):
        return image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)


class _RecordingProbe(BagelProbeMixin):
    def __init__(self, target_size):
        self.processing_class = SimpleNamespace(
            vae_image_transform=_FixedResizeTransform(target_size)
        )
        self.psnr_shapes = []
        self.ssim_shapes = []

    def _compute_psnr(self, img1, img2):
        self.psnr_shapes.append((tuple(img1.shape[-2:]), tuple(img2.shape[-2:])))
        return 0.0

    def _compute_ssim(self, img1, img2):
        self.ssim_shapes.append((tuple(img1.shape[-2:]), tuple(img2.shape[-2:])))
        return 0.0


def test_probe_variant_key_names_are_namespaced():
    probe = _RecordingProbe(target_size=(384, 512))

    metrics, _ = probe._build_probe_metrics(
        [
            {
                "sample_id": "probe_0",
                "outputs": [Image.new("RGB", (8, 8), color=(0, 0, 0))],
            }
        ],
        namespace="regular",
    )

    assert metrics == {"infer_probe/regular/avg_image_outputs": 1.0}

    variants = BagelProbeMixin._get_probe_variants(use_ema=True, compare_ema=True)
    assert variants == [("regular", False), ("ema", True)]


def test_single_ema_probe_keeps_legacy_metric_namespace():
    probe = _RecordingProbe(target_size=(384, 512))

    variants = BagelProbeMixin._get_probe_variants(use_ema=True, compare_ema=False)
    assert variants == [(None, True)]

    metrics, _ = probe._build_probe_metrics(
        [
            {
                "sample_id": "probe_0",
                "outputs": [Image.new("RGB", (8, 8), color=(0, 0, 0))],
            }
        ],
        namespace=None,
    )

    assert metrics == {"infer_probe/avg_image_outputs": 1.0}


def test_probe_metrics_align_ground_truth_to_training_vae_size(tmp_path):
    probe = _RecordingProbe(target_size=(384, 512))

    gen_image = Image.new("RGB", (512, 512), color=(64, 128, 192))
    gt_path = tmp_path / "gt.png"
    Image.new("RGB", (2048, 1536), color=(64, 128, 192)).save(gt_path)

    sample_result = {"outputs": [gen_image]}
    gt_comparison = probe._compute_gt_comparison(
        sample_result,
        [{"path": str(gt_path)}],
    )

    assert probe.psnr_shapes == [((384, 512), (384, 512))]
    assert probe.ssim_shapes == [((384, 512), (384, 512))]
    assert gt_comparison["gt_count"] == 1
    assert gt_comparison["gen_count"] == 1
    assert gt_comparison["per_pair"][0]["psnr"] == 0.0
    assert gt_comparison["per_pair"][0]["ssim"] == 0.0
