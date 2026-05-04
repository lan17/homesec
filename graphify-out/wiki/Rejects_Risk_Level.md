# Rejects Risk Level

> 25 nodes · cohesion 0.11

## Key Concepts

- **TestValidateRiskLevel** (13 connections) — `tests/homesec/test_enums.py`
- **_validate_risk_level()** (12 connections) — `src/homesec/models/enums.py`
- **enums.py** (10 connections) — `src/homesec/models/enums.py`
- **.test_accepts_risk_level_enum()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_accepts_valid_integer()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_accepts_valid_string()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_rejects_float()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_rejects_invalid_integer()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_rejects_invalid_string()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_rejects_list()** (3 connections) — `tests/homesec/test_enums.py`
- **.test_rejects_none()** (3 connections) — `tests/homesec/test_enums.py`
- **from_string()** (2 connections) — `src/homesec/models/enums.py`
- **_serialize_risk_level()** (2 connections) — `src/homesec/models/enums.py`
- **Tests for _validate_risk_level function.** (1 connections) — `tests/homesec/test_enums.py`
- **Should pass through RiskLevel unchanged.** (1 connections) — `tests/homesec/test_enums.py`
- **Should convert valid integers to RiskLevel.** (1 connections) — `tests/homesec/test_enums.py`
- **Should convert valid strings to RiskLevel.** (1 connections) — `tests/homesec/test_enums.py`
- **Should raise ValueError for out-of-range integers.** (1 connections) — `tests/homesec/test_enums.py`
- **Should raise ValueError for invalid strings.** (1 connections) — `tests/homesec/test_enums.py`
- **Should raise ValueError for None.** (1 connections) — `tests/homesec/test_enums.py`
- **Should raise ValueError for float.** (1 connections) — `tests/homesec/test_enums.py`
- **Should raise ValueError for list.** (1 connections) — `tests/homesec/test_enums.py`
- **Centralized enums for type safety and IDE support.** (1 connections) — `src/homesec/models/enums.py`
- **Validate and convert input to RiskLevel.      Accepts:         - RiskLevel enum** (1 connections) — `src/homesec/models/enums.py`
- **Serialize RiskLevel to lowercase string for config compatibility.** (1 connections) — `src/homesec/models/enums.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/models/enums.py`
- `tests/homesec/test_enums.py`

## Audit Trail

- EXTRACTED: 56 (75%)
- INFERRED: 19 (25%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*