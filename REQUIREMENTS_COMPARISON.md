# Requirements Comparison: requirements.txt vs Actual Environment

## Direct Comparison

### Requirements in `requirements.txt`:
```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
torch>=2.0.0
pyro-ppl>=1.8.0
oct2py>=5.0.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
graphviz>=0.20.0
```

### Actually Installed in `/kfs2/projects/hpcapps/nsawant/corrosion/env`:
```
numpy==2.2.6
pandas==2.3.3
scipy==1.15.2
torch==2.9.1
pyro-ppl==1.9.1+ab0491a
oct2py==5.8.0
matplotlib==3.10.8
scikit-learn==1.7.2
graphviz==0.21
```

## Package-by-Package Analysis

| Package | Required (min) | Installed | Satisfies? | Over by |
|---------|---------------|-----------|------------|---------|
| numpy | 1.20.0 | 2.2.6 | ✅ | 1.0.6 major versions |
| pandas | 1.3.0 | 2.3.3 | ✅ | 1.0.3 major versions |
| scipy | 1.7.0 | 1.15.2 | ✅ | 0.8.2 minor versions |
| torch | 2.0.0 | 2.9.1 | ✅ | 0.9.1 minor versions |
| pyro-ppl | 1.8.0 | 1.9.1 | ✅ | 0.1.1 minor versions |
| oct2py | 5.0.0 | 5.8.0 | ✅ | 0.8.0 minor versions |
| matplotlib | 3.4.0 | 3.10.8 | ✅ | 0.6.8 minor versions |
| scikit-learn | 1.0.0 | 1.7.2 | ✅ | 0.7.2 minor versions |
| graphviz | 0.20.0 | 0.21 | ✅ | 0.01 minor versions |

## Additional Dependency: pyro-api

**Note**: `pyro-ppl` automatically installed `pyro-api==0.1.2` as a dependency. This is not listed in requirements.txt but is required by pyro-ppl.

## Conclusion

✅ **All packages in `requirements.txt` are satisfied by the installed environment**

The environment is using significantly newer versions than the minimums specified:
- Most packages are 0.6-1.0 major/minor versions ahead
- No conflicts or missing dependencies
- `requirements.txt` correctly specifies minimum viable versions

## Recommendation

**Current `requirements.txt` is GOOD** because:
1. ✅ Uses `>=` constraints (allows flexibility)
2. ✅ Specifies reasonable minimum versions
3. ✅ All actually-used packages are present
4. ✅ Doesn't over-constrain versions (avoids dependency hell)

**Alternative approach** (not recommended unless needed):
Create a `requirements-frozen.txt` with exact versions for reproducibility:
```bash
pip freeze > requirements-frozen.txt
```

But for a research project, flexible minimum versions (current approach) are preferred over frozen exact versions.

---

*Generated: January 14, 2026*
