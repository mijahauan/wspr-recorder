"""Tests for configuration parsing."""

import pytest
import tempfile
import os
from wspr_recorder.config import (
    parse_frequency, freq_to_band_name, Config, ChannelDefaults,
    BandConfig, load_config,
)


class TestParseFrequency:
    """Tests for frequency parsing."""
    
    def test_plain_hz(self):
        assert parse_frequency("14095600") == 14095600
    
    def test_mhz_notation(self):
        assert parse_frequency("14m095600") == 14095600
        assert parse_frequency("7m038600") == 7038600
        assert parse_frequency("1m836600") == 1836600
    
    def test_khz_notation(self):
        assert parse_frequency("474k200") == 474200
        assert parse_frequency("136k000") == 136000
    
    def test_scientific_notation(self):
        assert parse_frequency("14.0956e6") == 14095600
    
    def test_whitespace(self):
        assert parse_frequency("  14095600  ") == 14095600
    
    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_frequency("invalid")


class TestFreqToBandName:
    """Tests for frequency to band name mapping."""
    
    def test_known_bands(self):
        assert freq_to_band_name(14095600) == "20"
        assert freq_to_band_name(7038600) == "40"
        assert freq_to_band_name(3568600) == "80"
        assert freq_to_band_name(474200) == "630"
    
    def test_eu_bands(self):
        assert freq_to_band_name(3592600) == "80eu"
        assert freq_to_band_name(5364700) == "60eu"
    
    def test_unknown_frequency(self):
        assert freq_to_band_name(12345678) == "12345678Hz"


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_valid_config(self):
        config = Config()
        config.frequencies = [14095600, 7038600]
        errors = config.validate()
        assert len(errors) == 0
    
    def test_no_frequencies(self):
        config = Config()
        errors = config.validate()
        assert any("No frequencies" in e for e in errors)
    
    def test_invalid_sample_format(self):
        config = Config()
        config.frequencies = [14095600]
        config.recorder.sample_format = "invalid"
        errors = config.validate()
        assert any("sample_format" in e for e in errors)

    def test_valid_timing_authorities(self):
        for authority in ("rtp", "fusion", "auto"):
            config = Config()
            config.frequencies = [14095600]
            config.timing.authority = authority
            errors = config.validate()
            assert len(errors) == 0

    def test_invalid_timing_authority(self):
        config = Config()
        config.frequencies = [14095600]
        config.timing.authority = "invalid"
        errors = config.validate()
        assert any("timing authority" in e for e in errors)


class TestBandConfig:
    """Tests for per-band configuration."""

    def test_default_modes(self):
        bc = BandConfig(frequency=14095600)
        assert bc.modes == ["W2"]

    def test_custom_modes(self):
        bc = BandConfig(frequency=14095600, modes=["W2", "F2", "F5"])
        assert bc.modes == ["W2", "F2", "F5"]

    def test_validate_valid(self):
        bc = BandConfig(frequency=14095600, modes=["W2", "F2", "F5", "F15", "F30"])
        assert bc.validate() == []

    def test_validate_invalid_mode(self):
        bc = BandConfig(frequency=14095600, modes=["W2", "X9"])
        errors = bc.validate()
        assert len(errors) == 1
        assert "X9" in errors[0]

    def test_validate_empty_modes(self):
        bc = BandConfig(frequency=14095600, modes=[])
        errors = bc.validate()
        assert any("no modes" in e for e in errors)

    def test_config_validate_catches_bad_band_mode(self):
        config = Config()
        config.frequencies = [14095600]
        config.bands = [BandConfig(frequency=14095600, modes=["INVALID"])]
        errors = config.validate()
        assert any("INVALID" in e for e in errors)


class TestGetBandConfig:
    def test_found(self):
        config = Config()
        config.bands = [
            BandConfig(frequency=14095600, modes=["W2", "F2"]),
            BandConfig(frequency=7038600, modes=["W2"]),
        ]
        bc = config.get_band_config(14095600)
        assert bc.modes == ["W2", "F2"]

    def test_not_found_returns_default(self):
        config = Config()
        config.bands = [BandConfig(frequency=14095600, modes=["W2", "F2"])]
        bc = config.get_band_config(7038600)
        assert bc.frequency == 7038600
        assert bc.modes == ["W2"]


class TestLoadConfigBandFormat:
    """Tests for loading [[band]] TOML format."""

    def _write_toml(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".toml")
        os.write(fd, content.encode())
        os.close(fd)
        return path

    def test_new_band_format(self):
        path = self._write_toml("""
[recorder]
output_dir = "/tmp/test"

[[band]]
frequency = "14095600"
modes = ["W2", "F2", "F5"]

[[band]]
frequency = "7038600"
modes = ["W2"]
""")
        try:
            config = load_config(path)
            assert len(config.bands) == 2
            assert config.bands[0].frequency == 14095600
            assert config.bands[0].modes == ["W2", "F2", "F5"]
            assert config.bands[1].frequency == 7038600
            assert config.bands[1].modes == ["W2"]
            # frequencies list also populated
            assert 14095600 in config.frequencies
            assert 7038600 in config.frequencies
        finally:
            os.unlink(path)

    def test_old_frequencies_format_backward_compat(self):
        path = self._write_toml("""
[recorder]
output_dir = "/tmp/test"

[frequencies]
bands = ["14095600", "7038600"]
""")
        try:
            config = load_config(path)
            assert len(config.bands) == 2
            # Old format defaults to W2 only
            assert config.bands[0].modes == ["W2"]
            assert config.bands[1].modes == ["W2"]
            assert config.frequencies == [14095600, 7038600]
        finally:
            os.unlink(path)

    def test_new_format_takes_precedence(self):
        """If both [[band]] and [frequencies] exist, [[band]] wins."""
        path = self._write_toml("""
[recorder]
output_dir = "/tmp/test"

[[band]]
frequency = "14095600"
modes = ["W2", "F2"]

[frequencies]
bands = ["7038600"]
""")
        try:
            config = load_config(path)
            # Only the [[band]] entry should be used
            assert len(config.bands) == 1
            assert config.bands[0].frequency == 14095600
        finally:
            os.unlink(path)

    def test_invalid_mode_in_band_raises(self):
        path = self._write_toml("""
[recorder]
output_dir = "/tmp/test"

[[band]]
frequency = "14095600"
modes = ["W2", "BOGUS"]
""")
        try:
            with pytest.raises(ValueError, match="BOGUS"):
                load_config(path)
        finally:
            os.unlink(path)
