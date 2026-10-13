_nvcc 13.0 · Tesla T4 (sm_75) · 2026-10-13_

| Kernel | debug-G | O0 | O1 | O2 | O3 | O3-nofma | O3-ptx75 | O3-fastmath | build-sm_80 | build-sm_90 |
|---|---|---|---|---|---|---|---|---|---|---|
| ArrayAddition | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔨 | 🔨 |
| ImageBlur | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | 🔨 | 🔨 |

✅ matches CPU reference and baseline · ⚠️ lenient config differs (expected) · ❌ cpu = GPU differs from CPU · ❌ diff = differs from baseline · 🔧❌ build failed · 💥 crashed · 🔨 built only (not run)
