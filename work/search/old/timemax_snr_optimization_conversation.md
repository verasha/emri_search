# Time Maximization and SNR Optimization Conversation Summary

## Date: 2025-12-17

## Key Problem
The F-statistic likelihood was degenerate because mode selection only captured ~30-56% of total SNR, violating the requirement that X·ρ = <d|h> (equation 11 from the paper).

## Initial State
- Original mode selection: ℓ=2, n ∈ [-5, 0) with only 5 mode groups
- SNR capture: ~55%
- X·ρ / <d|h> ratio: 0.31 (should be ~1.0)
- This caused the F-statistic to fail at breaking degeneracies

## Investigation Process

### 1. Time/Phase Maximization Discussion
- Initially explored using `timemax_inner` (FFT correlation trick) for time/phase marginalization
- Discovered this is only relevant for **extrinsic parameters** (coalescence time, phases)
- For **intrinsic parameter** searches (m1, m2, a, p0, e0), time-max doesn't help
- Intrinsic parameter changes alter waveform morphology, not just time/phase offsets

### 2. Root Cause Analysis
Found that the real issue was **insufficient mode coverage**:
```python
Standard <d|h>: 28.21
X·ρ from modes: 8.71
Ratio: 0.31  # Should be ~1.0!
```

Mode overlaps were good (< 0.005), so modes were orthogonal, but coverage was incomplete.

### 3. Systematic SNR Testing

Tested individual mode SNRs to find dominant contributions:

**Top modes by SNR:**
- (ℓ=3, m=3, n=-5): 1.65
- (ℓ=2, m=2, n=-5): 1.39
- (ℓ=2, m=2, n=-2): 1.36
- (ℓ=2, m=2, n=-1): 1.32
- (ℓ=3, m=3, n=-4): 1.26

**Key insight:** Need both ℓ=2 and ℓ=3, and likely ℓ=4,5 to reach 90% SNR capture.

### 4. Progressive Mode Expansion

| Configuration | SNR Capture | Notes |
|--------------|-------------|-------|
| ℓ=2, n ∈ [-5,0) | 55.67% | Original |
| ℓ=3, n ∈ [-5,0) | 43.76% | Worse alone |
| ℓ=2,3 mixed (5 groups) | 70.82% | Better but insufficient |
| ℓ=2,3,4 (5 groups) | 75.92% | Still not enough |
| ℓ=2,3,4,5, n ∈ [-6,1] (5 groups) | 88.75% | Close! |
| **ℓ=2,3,4,5, n ∈ [-7,1] (5 groups)** | **95.14%** | ✅ Success! |

## Final Optimal Solution

### Optimal Mode Groups (5 groups, 95.14% SNR capture)

```python
optimal_mode_groups = [
    # Group 1: n=-7,-6,-5 with ℓ=2,3,4,5 (SNR: 3.84)
    [(2,0,-7),(2,1,-7),(2,2,-7),(2,-1,-7),(2,-2,-7),
     (3,0,-7),(3,1,-7),(3,2,-7),(3,3,-7),(3,-1,-7),(3,-2,-7),(3,-3,-7),
     (4,0,-7),(4,1,-7),(4,2,-7),(4,3,-7),(4,4,-7),(4,-1,-7),(4,-2,-7),(4,-3,-7),(4,-4,-7),
     (5,0,-7),(5,1,-7),(5,2,-7),(5,3,-7),(5,4,-7),(5,5,-7),(5,-1,-7),(5,-2,-7),(5,-3,-7),(5,-4,-7),(5,-5,-7),
     (2,0,-6),(2,1,-6),(2,2,-6),(2,-1,-6),(2,-2,-6),
     (3,0,-6),(3,1,-6),(3,2,-6),(3,3,-6),(3,-1,-6),(3,-2,-6),(3,-3,-6),
     (4,0,-6),(4,1,-6),(4,2,-6),(4,3,-6),(4,4,-6),(4,-1,-6),(4,-2,-6),(4,-3,-6),(4,-4,-6),
     (5,0,-6),(5,1,-6),(5,2,-6),(5,3,-6),(5,4,-6),(5,5,-6),(5,-1,-6),(5,-2,-6),(5,-3,-6),(5,-4,-6),(5,-5,-6),
     (2,0,-5),(2,1,-5),(2,2,-5),(2,-1,-5),(2,-2,-5),
     (3,0,-5),(3,1,-5),(3,2,-5),(3,3,-5),(3,-1,-5),(3,-2,-5),(3,-3,-5),
     (4,0,-5),(4,1,-5),(4,2,-5),(4,3,-5),(4,4,-5),(4,-1,-5),(4,-2,-5),(4,-3,-5),(4,-4,-5),
     (5,0,-5),(5,1,-5),(5,2,-5),(5,3,-5),(5,4,-5),(5,5,-5),(5,-1,-5),(5,-2,-5),(5,-3,-5),(5,-4,-5),(5,-5,-5)],

    # Group 2: n=-4 with ℓ=2,3,4,5 (SNR: 2.01)
    [(2,0,-4),(2,1,-4),(2,2,-4),(2,-1,-4),(2,-2,-4),
     (3,0,-4),(3,1,-4),(3,2,-4),(3,3,-4),(3,-1,-4),(3,-2,-4),(3,-3,-4),
     (4,0,-4),(4,1,-4),(4,2,-4),(4,3,-4),(4,4,-4),(4,-1,-4),(4,-2,-4),(4,-3,-4),(4,-4,-4),
     (5,0,-4),(5,1,-4),(5,2,-4),(5,3,-4),(5,4,-4),(5,5,-4),(5,-1,-4),(5,-2,-4),(5,-3,-4),(5,-4,-4),(5,-5,-4)],

    # Group 3: n=-3 with ℓ=2,3,4,5 (SNR: 1.59)
    [(2,0,-3),(2,1,-3),(2,2,-3),(2,-1,-3),(2,-2,-3),
     (3,0,-3),(3,1,-3),(3,2,-3),(3,3,-3),(3,-1,-3),(3,-2,-3),(3,-3,-3),
     (4,0,-3),(4,1,-3),(4,2,-3),(4,3,-3),(4,4,-3),(4,-1,-3),(4,-2,-3),(4,-3,-3),(4,-4,-3),
     (5,0,-3),(5,1,-3),(5,2,-3),(5,3,-3),(5,4,-3),(5,5,-3),(5,-1,-3),(5,-2,-3),(5,-3,-3),(5,-4,-3),(5,-5,-3)],

    # Group 4: n=-2 with ℓ=2,3,4,5 (SNR: 1.47)
    [(2,0,-2),(2,1,-2),(2,2,-2),(2,-1,-2),(2,-2,-2),
     (3,0,-2),(3,1,-2),(3,2,-2),(3,3,-2),(3,-1,-2),(3,-2,-2),(3,-3,-2),
     (4,0,-2),(4,1,-2),(4,2,-2),(4,3,-2),(4,4,-2),(4,-1,-2),(4,-2,-2),(4,-3,-2),(4,-4,-2),
     (5,0,-2),(5,1,-2),(5,2,-2),(5,3,-2),(5,4,-2),(5,5,-2),(5,-1,-2),(5,-2,-2),(5,-3,-2),(5,-4,-2),(5,-5,-2)],

    # Group 5: n=-1,0,1 with ℓ=2,3,4,5 (SNR: 1.36)
    [(2,0,-1),(2,1,-1),(2,2,-1),(2,-1,-1),(2,-2,-1),
     (3,0,-1),(3,1,-1),(3,2,-1),(3,3,-1),(3,-1,-1),(3,-2,-1),(3,-3,-1),
     (4,0,-1),(4,1,-1),(4,2,-1),(4,3,-1),(4,4,-1),(4,-1,-1),(4,-2,-1),(4,-3,-1),(4,-4,-1),
     (5,0,-1),(5,1,-1),(5,2,-1),(5,3,-1),(5,4,-1),(5,5,-1),(5,-1,-1),(5,-2,-1),(5,-3,-1),(5,-4,-1),(5,-5,-1),
     (2,0,0),(2,1,0),(2,2,0),(2,-1,0),(2,-2,0),
     (3,0,0),(3,1,0),(3,2,0),(3,3,0),(3,-1,0),(3,-2,0),(3,-3,0),
     (4,0,0),(4,1,0),(4,2,0),(4,3,0),(4,4,0),(4,-1,0),(4,-2,0),(4,-3,0),(4,-4,0),
     (5,0,0),(5,1,0),(5,2,0),(5,3,0),(5,4,0),(5,5,0),(5,-1,0),(5,-2,0),(5,-3,0),(5,-4,0),(5,-5,0),
     (2,0,1),(2,1,1),(2,2,1),(2,-1,1),(2,-2,1),
     (3,0,1),(3,1,1),(3,2,1),(3,3,1),(3,-1,1),(3,-2,1),(3,-3,1)],
]
```

### Performance Metrics

**SNR Capture:**
- Group 1: 3.84
- Group 2: 2.01
- Group 3: 1.59
- Group 4: 1.47
- Group 5: 1.36
- **Total quadrature SNR: 5.03**
- **Full waveform SNR: 5.28**
- **Capture: 95.14%** ✅

**Runtime:**
- Group 1: 0.16s
- Group 2: 0.04s
- Group 3: 0.04s
- Group 4: 0.04s
- Group 5: 0.08s
- **Total: 0.38s per likelihood evaluation**

## Key Learnings

1. **F-statistic requires ~90% SNR capture** to work properly and break degeneracies
2. **Multi-ℓ coverage is essential** - single ℓ values are insufficient for EMRI waveforms
3. **Grouping modes by n-value** while including multiple ℓ values is an effective strategy
4. **Time/phase maximization** (Fourier trick) is only relevant for extrinsic parameter searches
5. **Mode orthogonality** (low overlap) is necessary but not sufficient - total coverage matters

## Next Steps

1. Fix `loglikealt.py` to set `self.selected_labels = self.mode_select` when using external mode selection
2. Test the F-statistic with optimal modes to verify degeneracy breaking
3. Run MCMC with the optimized mode selection
4. Monitor X·ρ / <d|h> ratio during searches to ensure it stays ~1.0

## Files Modified

- `/home/svu/e1498138/localgit/FEWNEW/work/check_timemax.ipynb` - Testing and validation
- `/home/svu/e1498138/localgit/FEWNEW/work/loglikealt.py` - Needs fix for `selected_labels`

## References

- Email notes: `email_response_notes.md`
- FFT correlation theory: `fft_correlation_and_mode_decomposition_reference.md`
- Paper equations: (9), (10), (11), (12), (14) for F-statistic formulation
