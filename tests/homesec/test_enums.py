"""Tests for centralized enums in models/enums.py."""

import pytest
from pydantic import BaseModel, ValidationError

from homesec.models.enums import (
    ClipStatus,
    EventType,
    RiskLevel,
    RiskLevelField,
    _validate_risk_level,
)


class TestEventType:
    """Tests for EventType StrEnum."""

    def test_all_event_types_are_strings(self) -> None:
        """EventType members should be strings."""
        for event_type in EventType:
            assert isinstance(event_type, str)
            assert isinstance(event_type.value, str)

    def test_event_type_equality_with_string(self) -> None:
        """EventType should be equal to its string value."""
        assert EventType.CLIP_RECORDED == "clip_recorded"
        assert EventType.VLM_COMPLETED == "vlm_completed"

    def test_event_type_count(self) -> None:
        """Should have exactly 16 event types."""
        assert len(EventType) == 16


class TestClipStatus:
    """Tests for ClipStatus StrEnum."""

    def test_all_statuses_are_strings(self) -> None:
        """ClipStatus members should be strings."""
        for status in ClipStatus:
            assert isinstance(status, str)

    def test_status_equality_with_string(self) -> None:
        """ClipStatus should be equal to its string value."""
        assert ClipStatus.QUEUED_LOCAL == "queued_local"
        assert ClipStatus.DONE == "done"

    def test_status_count(self) -> None:
        """Should have exactly 6 status values."""
        assert len(ClipStatus) == 6


class TestRiskLevel:
    """Tests for RiskLevel IntEnum."""

    def test_risk_level_ordering(self) -> None:
        """RiskLevel should support natural ordering."""
        assert RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_risk_level_comparison(self) -> None:
        """RiskLevel should support >= comparisons."""
        assert RiskLevel.HIGH >= RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM >= RiskLevel.MEDIUM
        assert not RiskLevel.LOW >= RiskLevel.MEDIUM

    def test_risk_level_values(self) -> None:
        """RiskLevel integer values should be 0-3."""
        assert RiskLevel.LOW == 0
        assert RiskLevel.MEDIUM == 1
        assert RiskLevel.HIGH == 2
        assert RiskLevel.CRITICAL == 3

    def test_str_returns_lowercase_name(self) -> None:
        """str(RiskLevel) should return lowercase name."""
        assert str(RiskLevel.LOW) == "low"
        assert str(RiskLevel.MEDIUM) == "medium"
        assert str(RiskLevel.HIGH) == "high"
        assert str(RiskLevel.CRITICAL) == "critical"

    def test_from_string_valid_lowercase(self) -> None:
        """from_string should accept lowercase strings."""
        assert RiskLevel.from_string("low") == RiskLevel.LOW
        assert RiskLevel.from_string("medium") == RiskLevel.MEDIUM
        assert RiskLevel.from_string("high") == RiskLevel.HIGH
        assert RiskLevel.from_string("critical") == RiskLevel.CRITICAL

    def test_from_string_valid_uppercase(self) -> None:
        """from_string should accept uppercase strings."""
        assert RiskLevel.from_string("LOW") == RiskLevel.LOW
        assert RiskLevel.from_string("MEDIUM") == RiskLevel.MEDIUM

    def test_from_string_valid_mixed_case(self) -> None:
        """from_string should accept mixed case strings."""
        assert RiskLevel.from_string("High") == RiskLevel.HIGH
        assert RiskLevel.from_string("CrItIcAl") == RiskLevel.CRITICAL

    def test_from_string_invalid_raises_value_error(self) -> None:
        """from_string should raise ValueError for invalid strings."""
        with pytest.raises(ValueError) as exc_info:
            RiskLevel.from_string("invalid")
        assert "Invalid risk level 'invalid'" in str(exc_info.value)
        assert "low, medium, high, critical" in str(exc_info.value)

    def test_from_string_empty_raises_value_error(self) -> None:
        """from_string should raise ValueError for empty string."""
        with pytest.raises(ValueError):
            RiskLevel.from_string("")


class TestValidateRiskLevel:
    """Tests for _validate_risk_level function."""

    def test_accepts_risk_level_enum(self) -> None:
        """Should pass through RiskLevel unchanged."""
        assert _validate_risk_level(RiskLevel.HIGH) == RiskLevel.HIGH

    def test_accepts_valid_integer(self) -> None:
        """Should convert valid integers to RiskLevel."""
        assert _validate_risk_level(0) == RiskLevel.LOW
        assert _validate_risk_level(1) == RiskLevel.MEDIUM
        assert _validate_risk_level(2) == RiskLevel.HIGH
        assert _validate_risk_level(3) == RiskLevel.CRITICAL

    def test_accepts_valid_string(self) -> None:
        """Should convert valid strings to RiskLevel."""
        assert _validate_risk_level("low") == RiskLevel.LOW
        assert _validate_risk_level("HIGH") == RiskLevel.HIGH

    def test_rejects_invalid_integer(self) -> None:
        """Should raise ValueError for out-of-range integers."""
        with pytest.raises(ValueError):
            _validate_risk_level(4)
        with pytest.raises(ValueError):
            _validate_risk_level(-1)

    def test_rejects_invalid_string(self) -> None:
        """Should raise ValueError for invalid strings."""
        with pytest.raises(ValueError):
            _validate_risk_level("not_a_level")

    def test_rejects_none(self) -> None:
        """Should raise ValueError for None."""
        with pytest.raises(ValueError) as exc_info:
            _validate_risk_level(None)
        assert "Cannot convert NoneType to RiskLevel" in str(exc_info.value)

    def test_rejects_float(self) -> None:
        """Should raise ValueError for float."""
        with pytest.raises(ValueError) as exc_info:
            _validate_risk_level(1.5)
        assert "Cannot convert float to RiskLevel" in str(exc_info.value)

    def test_rejects_list(self) -> None:
        """Should raise ValueError for list."""
        with pytest.raises(ValueError) as exc_info:
            _validate_risk_level(["low"])
        assert "Cannot convert list to RiskLevel" in str(exc_info.value)


class TestRiskLevelFieldPydantic:
    """Tests for RiskLevelField in Pydantic models."""

    def test_pydantic_model_accepts_string(self) -> None:
        """Pydantic model with RiskLevelField should accept strings."""

        class TestModel(BaseModel):
            level: RiskLevelField

        model = TestModel(level="high")
        assert model.level == RiskLevel.HIGH

    def test_pydantic_model_accepts_enum(self) -> None:
        """Pydantic model with RiskLevelField should accept RiskLevel."""

        class TestModel(BaseModel):
            level: RiskLevelField

        model = TestModel(level=RiskLevel.MEDIUM)
        assert model.level == RiskLevel.MEDIUM

    def test_pydantic_model_accepts_int(self) -> None:
        """Pydantic model with RiskLevelField should accept integers."""

        class TestModel(BaseModel):
            level: RiskLevelField

        model = TestModel(level=2)
        assert model.level == RiskLevel.HIGH

    def test_pydantic_model_rejects_invalid(self) -> None:
        """Pydantic model should reject invalid values."""

        class TestModel(BaseModel):
            level: RiskLevelField

        with pytest.raises(ValidationError):
            TestModel(level="invalid")

    def test_pydantic_serialization_to_string(self) -> None:
        """model_dump should serialize RiskLevel to string."""

        class TestModel(BaseModel):
            level: RiskLevelField

        model = TestModel(level=RiskLevel.HIGH)
        dumped = model.model_dump()
        assert dumped["level"] == "high"
        assert isinstance(dumped["level"], str)

    def test_pydantic_json_serialization(self) -> None:
        """model_dump(mode='json') should serialize RiskLevel to string."""

        class TestModel(BaseModel):
            level: RiskLevelField

        model = TestModel(level=RiskLevel.CRITICAL)
        json_data = model.model_dump(mode="json")
        assert json_data["level"] == "critical"

    def test_pydantic_round_trip(self) -> None:
        """model_dump then model_validate should preserve value."""

        class TestModel(BaseModel):
            level: RiskLevelField

        original = TestModel(level=RiskLevel.MEDIUM)
        dumped = original.model_dump()
        restored = TestModel.model_validate(dumped)
        assert restored.level == original.level
        assert restored.level == RiskLevel.MEDIUM

    def test_pydantic_optional_field(self) -> None:
        """RiskLevelField should work with Optional."""

        class TestModel(BaseModel):
            level: RiskLevelField | None = None

        model_none = TestModel()
        assert model_none.level is None

        model_value = TestModel(level="low")
        assert model_value.level == RiskLevel.LOW
