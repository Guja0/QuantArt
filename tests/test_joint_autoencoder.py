"""Verify joint gradients, mode switching, and full checkpoint compatibility."""
import copy
import unittest

import torch
import torch.nn.functional as F

from finetune_artwork_decoder import ContinuousArtworkAE, edge_loss, ema_update


class JointAutoencoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        self.model = ContinuousArtworkAE(dict(double_z=False, z_channels=8, resolution=32,
            in_channels=3, out_ch=3, ch=8, ch_mult=[1, 2], num_res_blocks=1,
            attn_resolutions=[16], dropout=0.0), 8)

    def test_joint_update_and_roundtrip(self):
        m = self.model
        params = m.configure_train_part("both")
        self.assertEqual(len(params), len(list(m.parameters())))
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        m.set_train_mode("both")
        before = {k: v.clone() for k, v in m.state_dict().items()}
        ema = copy.deepcopy(m).requires_grad_(False)
        x = torch.randn(2, 3, 32, 32)
        optimizer = torch.optim.Adam(params, lr=1e-4)
        y = m.decode(m.encode(x))
        loss = F.l1_loss(y, x) + 0.2 * edge_loss(y, x).mean()
        loss.backward()
        for module in (m.encoder_real, m.quant_conv_real, m.post_quant_conv, m.decoder):
            self.assertTrue(module.training)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        optimizer.step()
        ema_update(ema, m, 0, 0.999)
        for prefix in ("encoder_real.", "quant_conv_real.", "post_quant_conv.", "decoder."):
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in m.state_dict().items() if k.startswith(prefix)))
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in ema.state_dict().items() if k.startswith(prefix)))
        reloaded = copy.deepcopy(self.model)
        reloaded.load_state_dict(ema.state_dict(), strict=True)
        ema.eval(); reloaded.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(ema.decode(ema.encode(x)), reloaded.decode(reloaded.encode(x))))
        self.assertEqual(set(ema.state_dict()), set(ema.encoder_state_dict()) | set(ema.decoder_state_dict()))

    def test_single_sides_and_return_to_joint(self):
        m = self.model
        m.configure_train_part("encoder")
        m.set_train_mode("encoder")
        self.assertFalse(m.decoder.training)
        self.assertFalse(any(p.requires_grad for p in m.decoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.encoder_parameters()))
        m.configure_train_part("decoder")
        m.set_train_mode("decoder")
        self.assertFalse(m.encoder_real.training)
        self.assertFalse(any(p.requires_grad for p in m.encoder_parameters()))
        self.assertTrue(all(p.requires_grad for p in m.decoder_parameters()))
        m.configure_train_part("both")
        m.set_train_mode("both")
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        with self.assertRaises(ValueError):
            m.configure_train_part("both", ["decoder.conv_out"])


if __name__ == "__main__":
    unittest.main()
