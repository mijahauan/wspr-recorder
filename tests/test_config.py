"""Tests for configuration parsing."""

import pytest
from wspr_recorder.config import parse_frequency, freq_to_band_name, Config, ChannelDefaults


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
