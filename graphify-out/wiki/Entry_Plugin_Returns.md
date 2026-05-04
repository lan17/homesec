# Entry Plugin Returns

> 36 nodes · cohesion 0.08

## Key Concepts

- **TestLoadPluginFromEntryPoint** (9 connections) — `tests/homesec/test_plugin_utils.py`
- **._make_entry_point()** (8 connections) — `tests/homesec/test_plugin_utils.py`
- **load_plugin_from_entry_point()** (8 connections) — `src/homesec/plugins/utils.py`
- **iter_entry_points()** (6 connections) — `src/homesec/plugins/utils.py`
- **_FakeEntryPoints** (5 connections) — `tests/homesec/test_plugin_utils.py`
- **TestIterEntryPoints** (5 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_returns_empty_for_unknown_group()** (5 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_returns_entry_points_from_select()** (5 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_loads_direct_plugin_instance()** (5 connections) — `tests/homesec/test_plugin_utils.py`
- **test_plugin_utils.py** (5 connections) — `tests/homesec/test_plugin_utils.py`
- **DummyPlugin** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_returns_entry_points_from_mapping()** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_accepts_subclass_instances()** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_factory_can_return_subclass()** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_loads_plugin_from_factory()** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_raises_for_wrong_type()** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **.test_raises_when_factory_returns_wrong_type()** (4 connections) — `tests/homesec/test_plugin_utils.py`
- **utils.py** (3 connections) — `src/homesec/plugins/utils.py`
- **.__init__()** (1 connections) — `tests/homesec/test_plugin_utils.py`
- **Tests for plugin utility functions.** (1 connections) — `tests/homesec/test_plugin_utils.py`
- **Create a mock entry point that returns the given value on load().** (1 connections) — `tests/homesec/test_plugin_utils.py`
- **Loads plugin when entry point returns instance directly.** (1 connections) — `tests/homesec/test_plugin_utils.py`
- **Loads plugin when entry point returns a factory callable.** (1 connections) — `tests/homesec/test_plugin_utils.py`
- **Raises TypeError when entry point returns wrong type.** (1 connections) — `tests/homesec/test_plugin_utils.py`
- **Raises TypeError when factory returns wrong type.** (1 connections) — `tests/homesec/test_plugin_utils.py`
- *... and 11 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/plugins/utils.py`
- `tests/homesec/test_plugin_utils.py`

## Audit Trail

- EXTRACTED: 91 (83%)
- INFERRED: 19 (17%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*