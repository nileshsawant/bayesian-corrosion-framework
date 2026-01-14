# Environment Status

## Comparison: requirements.txt vs Installed Packages

### Current Environment: `/kfs2/projects/hpcapps/nsawant/corrosion/env`

| Package | Required Version | Installed Version | Status |
|---------|-----------------|-------------------|--------|
| numpy | >=1.20.0 | 2.2.6 | ✅ |
| pandas | >=1.3.0 | 2.3.3 | ✅ |
| scipy | >=1.7.0 | 1.15.2 | ✅ |
| torch | >=2.0.0 | 2.9.1 | ✅ |
| pyro-ppl | >=1.8.0 | 1.9.1+ab0491a | ✅ |
| oct2py | >=5.0.0 | 5.8.0 | ✅ |
| matplotlib | >=3.4.0 | 3.10.8 | ✅ |
| scikit-learn | >=1.0.0 | 1.7.2 | ✅ |
| graphviz | >=0.20.0 | 0.21 | ✅ |

### Additional Installed Packages (Selected)

Packages installed in the environment but not listed in requirements.txt:

- **ipython** (8.37.0) - Interactive Python shell
- **jupyter_client** (8.7.0) - Jupyter kernel client
- **ipykernel** (7.1.0) - IPython kernel for Jupyter
- **debugpy** (1.8.18) - Debugger for Python
- **jinja2** (3.1.6) - Template engine (used by Jupyter)

These are development tools and are not required for production use of the framework.

## Summary

✅ **All required packages are installed and meet minimum version requirements**

The environment at `/kfs2/projects/hpcapps/nsawant/corrosion/env` is fully compatible with the updated `requirements.txt`.

## Recent Update

- **Added**: `graphviz>=0.20.0` (required for `generate_flowchart.py` script)

## Installation Command

To recreate this environment:

```bash
pip install -r requirements.txt
```

Or to update existing environment:

```bash
pip install --upgrade -r requirements.txt
```

---

*Last updated: January 14, 2026*
