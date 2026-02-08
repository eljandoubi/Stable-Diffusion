# Stable Diffusion

## LPIPS Evaluation on Validation Set

This section reports validation accuracy for LPIPS across multiple datasets, comparing a reproduced implementation against the original LPIPS.

### Results

| Dataset        | LPIPS Reproduction | Original LPIPS |
|---------------|-------------------:|---------------:|
| CNN           | 0.826              | 0.821          |
| TRADITIONAL   | 0.769              | 0.732          |
| COLOR         | 0.623              | 0.615          |
| DEBLUR        | 0.594              | 0.594          |
| FRAMEINTERP   | 0.629              | 0.622          |
| SUPERRES      | 0.694              | 0.695          |

### Notes

- The reproduced LPIPS closely matches the original implementation across all datasets.
- Performance differences are minor and within expected variance.
- Exact parity is observed on the **DEBLUR** dataset.
