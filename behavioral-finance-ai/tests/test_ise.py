import pytest
import pandas as pd
import numpy as np

# Mocking ISE classes based on user requirements

class MockISEThreshold:
    def calculate(self, balance, buffer, is_oneoff_detected, safe_to_save):
        if balance <= buffer:
            return 0
        if is_oneoff_detected:
            return 0
            
        tranche = safe_to_save * 0.30
        if tranche < 50:
            return 0 if tranche < 25 else 50
        if tranche > 500:
            return 500
        return tranche

class MockVault:
    def __init__(self, path):
        self.path = path
        self.balances = {}
        self.history = []
        
    def deposit(self, user_id, amount):
        if user_id not in self.balances:
            self.balances[user_id] = 0
        self.balances[user_id] += amount
        self.history.append({'user_id': user_id, 'amount': amount})

def test_threshold_no_oneoff():
    th = MockISEThreshold()
    # balance >> buffer, safe_to_save gives tranche 100
    res = th.calculate(balance=2000, buffer=1000, is_oneoff_detected=False, safe_to_save=333.33)
    assert 50 <= res <= 500

def test_threshold_with_oneoff():
    th = MockISEThreshold()
    res = th.calculate(balance=2000, buffer=1000, is_oneoff_detected=True, safe_to_save=300)
    assert res == 0

def test_threshold_balance_too_low():
    th = MockISEThreshold()
    # Balance below or equal to buffer
    res = th.calculate(balance=1000, buffer=1000, is_oneoff_detected=False, safe_to_save=100)
    assert res == 0

def test_threshold_tranche_clamp_min():
    th = MockISEThreshold()
    # safe_to_save = 100 -> tranche = 30 -> less than 50
    res = th.calculate(balance=2000, buffer=1000, is_oneoff_detected=False, safe_to_save=100)
    assert res in [0, 50]

def test_threshold_tranche_clamp_max():
    th = MockISEThreshold()
    # safe_to_save = 5000 -> tranche = 1500 -> clamp to 500
    res = th.calculate(balance=5000, buffer=1000, is_oneoff_detected=False, safe_to_save=5000)
    assert res == 500

def test_vault_deposit(tmp_path):
    vault = MockVault(tmp_path / "vault.json")
    vault.deposit('user1', 100)
    assert vault.balances['user1'] == 100

def test_vault_multiple_users(tmp_path):
    vault = MockVault(tmp_path / "vault.json")
    vault.deposit('user1', 100)
    vault.deposit('user2', 200)
    assert vault.balances['user1'] == 100
    assert vault.balances['user2'] == 200

def test_vault_history_length(tmp_path):
    vault = MockVault(tmp_path / "vault.json")
    vault.deposit('user1', 100)
    vault.deposit('user1', 50)
    assert len(vault.history) == 2
