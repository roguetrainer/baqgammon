#!/usr/bin/env python3
"""
Quantum Backgammon - Variant 1 Prototype
Simple playable demo of quantum dice with delayed measurement
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict

class QuantumDice:
    """Quantum dice in superposition of all 36 outcomes"""
    
    def __init__(self):
        # Equal superposition: each outcome has amplitude 1/6 per die
        self.amplitudes = np.ones((6, 6), dtype=complex) / 6.0
        self.measured = False
        self.result: Optional[Tuple[int, int]] = None
        
    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution |amplitude|²"""
        return np.abs(self.amplitudes) ** 2
    
    def show_superposition(self):
        """Display current quantum state"""
        probs = self.get_probabilities()
        print("\n" + "="*60)
        print("⚛️  QUANTUM DICE STATE ⚛️".center(60))
        print("="*60)
        print("\nAll 36 outcomes in SUPERPOSITION:")
        print("\n    ", end="")
        for j in range(6):
            print(f"  {j+1:2d}  ", end="")
        print()
        print("   " + "-"*42)
        
        for i in range(6):
            print(f" {i+1} |", end="")
            for j in range(6):
                prob = probs[i, j]
                print(f" {prob*100:4.1f}%", end="")
            print()
        
        print("\n💡 Each outcome: ~2.78% probability")
        print("="*60 + "\n")
    
    def measure(self) -> Tuple[int, int]:
        """Collapse the quantum state (measurement)"""
        if self.measured:
            return self.result
        
        print("\n⚡ MEASURING QUANTUM STATE... ⚡")
        time.sleep(0.5)
        
        # Dramatic countdown
        for i in range(3, 0, -1):
            print(f"   {i}...", end="", flush=True)
            time.sleep(0.4)
        print("\n")
        
        # Randomly choose based on probability distribution
        probs = self.get_probabilities().flatten()
        outcome_idx = np.random.choice(36, p=probs)
        die1 = outcome_idx // 6 + 1
        die2 = outcome_idx % 6 + 1
        
        self.measured = True
        self.result = (die1, die2)
        
        # Animate collapse
        print("🌊 WAVE FUNCTION COLLAPSING... 🌊")
        time.sleep(0.5)
        
        # Collapse amplitudes
        self.amplitudes = np.zeros((6, 6), dtype=complex)
        self.amplitudes[die1-1, die2-1] = 1.0
        
        print(f"\n✨ COLLAPSED TO: {die1}-{die2} ✨\n")
        time.sleep(0.5)
        
        return self.result


class SimpleBoard:
    """Simplified backgammon board for demo"""
    
    def __init__(self):
        # Just track a few key positions for demo
        # Position -> (player, count)
        self.positions: Dict[int, Tuple[int, int]] = {
            8: (0, 2),   # Player 0 has 2 checkers on point 8
            13: (0, 5),  # Player 0 has 5 checkers on point 13
            6: (1, 5),   # Player 1 has 5 checkers on point 6
            17: (1, 3),  # Player 1 has 3 checkers on point 17
        }
        
    def show(self, player_perspective: int = 0):
        """Display simplified board"""
        print("\n📊 BOARD STATE:")
        print("-" * 50)
        for pos in sorted(self.positions.keys()):
            player, count = self.positions[pos]
            if player == player_perspective:
                symbol = "●"
            else:
                symbol = "○"
            print(f"  Point {pos:2d}: {symbol * count}")
        print("-" * 50 + "\n")


class QuantumBackgammonVariant1:
    """Playable demo of Variant 1: Quantum Dice"""
    
    def __init__(self):
        self.board = SimpleBoard()
        self.current_player = 0
        self.dice: Optional[QuantumDice] = None
        
    def play_turn(self, player: int):
        """Play one turn"""
        print("\n" + "="*60)
        print(f"PLAYER {player}'s TURN".center(60))
        print("="*60 + "\n")
        
        self.board.show(player)
        
        # Roll quantum dice
        self.dice = QuantumDice()
        self.dice.show_superposition()
        
        # Player decision: measure now or delay?
        print("⚛️  QUANTUM DECISION POINT ⚛️\n")
        print("Options:")
        print("  [M] Measure dice NOW (see definite outcome)")
        print("  [D] Delay measurement (keep opponent uncertain)")
        print()
        
        choice = input("Your choice (M/D): ").strip().upper()
        
        if choice == 'M':
            # Immediate measurement
            result = self.dice.measure()
            print(f"\n✓ You rolled {result[0]}-{result[1]}")
            print("  Opponent knows exactly what you rolled.")
            print("  You can plan your move precisely.")
            
        else:
            # Delayed measurement
            print("\n🌀 KEEPING DICE IN SUPERPOSITION! 🌀")
            print("\n💡 Strategic implications:")
            print("  ✓ Opponent uncertain about your position")
            print("  ✓ You maintain flexibility")
            print("  ⚠️  But you can't plan precisely either")
            print("\nDeclare your intended move:")
            print("  (e.g., 'Move as if I rolled 6-5')")
            
            intended = input("\nIntended roll (e.g., '6-5'): ").strip()
            print(f"\n✓ You declared: 'Moving as if {intended}'")
            print("  Your checkers enter SUPERPOSITION")
            print("  Opponent sees probability distribution")
            
            input("\nPress ENTER to continue to end of turn...")
            
            # Must measure at end of turn
            print("\n⚠️  End of turn - FORCED MEASUREMENT")
            result = self.dice.measure()
            
            if intended.replace('-', '') == f"{result[0]}{result[1]}" or \
               intended.replace('-', '') == f"{result[1]}{result[0]}":
                print(f"🎉 Lucky! You got your intended roll!")
            else:
                print(f"😅 Dice collapsed to {result[0]}-{result[1]}")
                print(f"   Different from your intended {intended}")
    
    def calculate_von_neumann_entropy(self) -> float:
        """Calculate uncertainty in dice state"""
        if self.dice is None or self.dice.measured:
            return 0.0
        
        probs = self.dice.get_probabilities().flatten()
        # Remove zeros for numerical stability
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log2(probs))
    
    def show_quantum_metrics(self):
        """Display quantum information metrics"""
        entropy = self.calculate_von_neumann_entropy()
        max_entropy = np.log2(36)  # Maximum for uniform distribution
        
        print("\n📈 QUANTUM METRICS:")
        print(f"  Von Neumann Entropy: {entropy:.2f} bits")
        print(f"  Maximum Entropy: {max_entropy:.2f} bits")
        print(f"  Uncertainty: {entropy/max_entropy:.1%}")
        
        if entropy > 4.5:
            print("  ⚠️  HIGH UNCERTAINTY - consider measuring")
        elif entropy < 1.0:
            print("  ✓ LOW UNCERTAINTY - good information state")


def demo_comparison():
    """Demonstrate classical vs quantum dice"""
    print("\n" + "="*60)
    print("CLASSICAL vs QUANTUM DICE".center(60))
    print("="*60 + "\n")
    
    print("CLASSICAL DICE:")
    print("  • Roll dice: they land on specific values")
    print("  • You just don't see them yet")
    print("  • Hidden but DEFINITE")
    print("  • Example: Die is 🎲-5, you just don't know it")
    print()
    
    print("QUANTUM DICE:")
    print("  • Roll dice: they're in ALL states at once")
    print("  • Not hidden - genuinely SUPERPOSED")
    print("  • No definite value until measured")
    print("  • Example: Die is (🎲-1 + 🎲-2 + ... + 🎲-6)/√6")
    print("\n" + "="*60 + "\n")
    
    input("Press ENTER to continue...")


def tutorial():
    """Interactive tutorial"""
    print("\n" + "="*60)
    print("QUANTUM BACKGAMMON - VARIANT 1 TUTORIAL".center(60))
    print("="*60 + "\n")
    
    print("Welcome to Quantum Backgammon!")
    print("\nIn this variant, dice exist in quantum superposition.")
    print("You can choose WHEN to measure (collapse) them.")
    print()
    
    input("Press ENTER to learn about superposition...")
    
    print("\n💡 SUPERPOSITION")
    print("-" * 50)
    print("When you roll quantum dice, they don't land on a")
    print("specific number - they're in ALL 36 outcomes at once!")
    print()
    print("This isn't just 'hidden' like classical dice.")
    print("It's a genuine quantum mechanical superposition.")
    print()
    
    input("Press ENTER to learn about measurement...")
    
    print("\n💡 MEASUREMENT")
    print("-" * 50)
    print("When you measure the dice, the superposition 'collapses'")
    print("to one definite outcome. This is irreversible!")
    print()
    print("Strategic choice:")
    print("  • Measure early = know your roll, opponent knows too")
    print("  • Delay measurement = keep opponent uncertain, but")
    print("    you can't plan precisely either")
    print()
    
    input("Press ENTER to learn about strategy...")
    
    print("\n💡 QUANTUM STRATEGY")
    print("-" * 50)
    print("Delaying measurement creates a probability cloud:")
    print("  • Your checkers are in superposition across points")
    print("  • Opponent can't plan defense precisely")
    print("  • But you can't plan offense precisely either!")
    print()
    print("Trade-off between:")
    print("  • Information (knowing your position)")
    print("  • Uncertainty (keeping opponent guessing)")
    print()
    
    input("Press ENTER to start game...")


def main():
    """Main game loop"""
    print("\n" + "█"*60)
    print("QUANTUM BACKGAMMON - VARIANT 1".center(60))
    print("Quantum Dice with Delayed Measurement".center(60))
    print("█"*60)
    
    # Show classical vs quantum comparison
    demo_comparison()
    
    # Tutorial
    show_tutorial = input("Show tutorial? (y/n): ").strip().lower()
    if show_tutorial == 'y':
        tutorial()
    
    # Create game
    game = QuantumBackgammonVariant1()
    
    # Play a few turns
    print("\n🎮 Let's play a few turns!")
    print("(Simplified demo - just experiencing quantum dice)\n")
    
    for turn in range(2):
        game.play_turn(turn % 2)
        game.show_quantum_metrics()
        
        if turn < 1:
            input("\n[Next player's turn - Press ENTER]")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!".center(60))
    print("="*60)
    print("\n✨ You've experienced quantum superposition,")
    print("   measurement, and the strategy of delayed measurement!")
    print("\n🔬 Key quantum concepts demonstrated:")
    print("   • Superposition (dice in all states at once)")
    print("   • Measurement (collapse to definite state)")
    print("   • Strategic timing (when to measure)")
    print("   • Von Neumann entropy (quantifying uncertainty)")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
