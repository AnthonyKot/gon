package neuralnet

import (
	"math"
	"testing"
)

// Helper function for comparing floats with a tolerance
func floatEquals(a, b, tolerance float32) bool {
	return float32(math.Abs(float64(a-b))) < tolerance
}

// TestCalculateCurrentLr function
func TestCalculateCurrentLr(t *testing.T) {
	tests := []struct {
		description        string
		params             Params
		currentGlobalStep  int
		totalTrainingSteps int
		expectedLr         float32
	}{
		// 1. Warm-up Phase
		{
			description:        "Linear warm-up from 0 to 0.1 over 100 steps, at step 0",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine"},
			currentGlobalStep:  0,
			totalTrainingSteps: 1000,
			expectedLr:         0.0,
		},
		{
			description:        "Linear warm-up from 0 to 0.1 over 100 steps, at step 50",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine"},
			currentGlobalStep:  50,
			totalTrainingSteps: 1000,
			expectedLr:         0.05,
		},
		{
			description:        "Linear warm-up from 0 to 0.1 over 100 steps, at step 99",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine"},
			currentGlobalStep:  99,
			totalTrainingSteps: 1000,
			expectedLr:         0.099,
		},
		{
			description:        "Linear warm-up, at WarmupSteps boundary (step 100)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine", DecaySteps: 900}, // Cosine decay starts here
			currentGlobalStep:  100,
			totalTrainingSteps: 1000, // Warmup (100) + Decay (900)
			expectedLr:         0.1,  // At stepAfterWarmup = 0 for cosine, factor is 1.0
		},

		// 2. Post-Warm-up - "cosine" Schedule
		{
			description:        "Cosine decay, 0 warmup, start of decay",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 0, LrSchedule: "cosine", DecaySteps: 1000},
			currentGlobalStep:  0,
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},
		{
			description:        "Cosine decay, 0 warmup, mid decay (500/1000 steps)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 0, LrSchedule: "cosine", DecaySteps: 1000},
			currentGlobalStep:  500,
			totalTrainingSteps: 1000,
			expectedLr:         0.05, // 0.1 * 0.5 * (1 + cos(pi * 500/1000)) = 0.1 * 0.5 * (1+0) = 0.05
		},
		{
			description:        "Cosine decay, 0 warmup, end of decay (1000/1000 steps)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 0, LrSchedule: "cosine", DecaySteps: 1000},
			currentGlobalStep:  1000,
			totalTrainingSteps: 1000,
			expectedLr:         0.0, // 0.1 * 0.5 * (1 + cos(pi*1)) = 0.1 * 0.5 * (1-1) = 0
		},
		{
			description:        "Cosine decay with warmup, start of decay (step 100, after 100 warmup steps)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine", DecaySteps: 900},
			currentGlobalStep:  100, // stepAfterWarmup = 0
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},
		{
			description:        "Cosine decay with warmup, mid decay (step 550: 100 warmup + 450 decay)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine", DecaySteps: 900},
			currentGlobalStep:  550, // stepAfterWarmup = 450
			totalTrainingSteps: 1000,
			expectedLr:         0.05, // 450/900 = 0.5
		},
		{
			description:        "Cosine decay, DecaySteps=0 (decay over all post-warmup steps)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine", DecaySteps: 0},
			currentGlobalStep:  100, // stepAfterWarmup = 0. decayStepsToUse = 1000-100 = 900
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},
		{
			description:        "Cosine decay, DecaySteps=0, mid decay",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine", DecaySteps: 0},
			currentGlobalStep:  550, // stepAfterWarmup = 450. decayStepsToUse = 1000-100=900
			totalTrainingSteps: 1000,
			expectedLr:         0.05,
		},

		// 3. Post-Warm-up - "exponential" Schedule
		{
			description:        "Exponential schedule, 0 warmup, step 0",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 0, LrSchedule: "exponential", Decay: 0.95},
			currentGlobalStep:  0,
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},
		{
			description:        "Exponential schedule with warmup, step 100 (post-warmup)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "exponential", Decay: 0.95},
			currentGlobalStep:  100,
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},
		{
			description:        "Exponential schedule with warmup, step 500 (well past-warmup)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "exponential", Decay: 0.95},
			currentGlobalStep:  500,
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},

		// 4. Post-Warm-up - "none" Schedule
		{
			description:        "None schedule, 0 warmup, step 0",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 0, LrSchedule: "none"},
			currentGlobalStep:  0,
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},
		{
			description:        "None schedule with warmup, step 100 (post-warmup)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "none"},
			currentGlobalStep:  100,
			totalTrainingSteps: 1000,
			expectedLr:         0.1,
		},

		// 5. Edge Cases
		{
			description:        "Zero warmup steps, step 0, cosine", // Essentially same as: "Cosine decay, 0 warmup, start of decay"
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 0, LrSchedule: "cosine", DecaySteps: 100},
			currentGlobalStep:  0,
			totalTrainingSteps: 100,
			expectedLr:         0.1,
		},
		{
			description:        "Cosine decay, DecaySteps <= 0 (uses totalTrainingSteps - warmup), non-zero WarmupSteps, start of decay",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 10, LrSchedule: "cosine", DecaySteps: 0},
			currentGlobalStep:  10, // stepAfterWarmup = 0. decayStepsToUse = 100-10 = 90
			totalTrainingSteps: 100,
			expectedLr:         0.1,
		},
		{
			description:        "Cosine decay, DecaySteps explicit, current step > warmup + decay steps",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 10, LrSchedule: "cosine", DecaySteps: 50},
			currentGlobalStep:  100, // stepAfterWarmup = 90. decayStepsToUse = 50. cosineFrac = 90/50 > 1.0
			totalTrainingSteps: 1000,
			expectedLr:         0.0, // cosineFrac clamped to 1.0, factor is 0.5 * (1 + cos(pi)) = 0
		},
		{
			description:        "Cosine decay, totalTrainingSteps <= WarmupSteps (no decay phase)",
			params:             Params{InitialLr: 0.0, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "cosine", DecaySteps: 0},
			currentGlobalStep:  100, // stepAfterWarmup = 0. decayStepsToUse = 100-100 = 0
			totalTrainingSteps: 100,
			expectedLr:         0.1, // decayStepsToUse is 0, so returns TargetLr
		},
		{
			description:        "Warmup from InitialLr != 0",
			params:             Params{InitialLr: 0.01, TargetLr: 0.1, WarmupSteps: 100, LrSchedule: "none"},
			currentGlobalStep:  50,
			totalTrainingSteps: 1000,
			expectedLr:         0.055, // 0.01 + (0.1-0.01)*0.5 = 0.01 + 0.045
		},
	}

	tolerance := float32(1e-6) // Tolerance for float comparisons

	for _, tt := range tests {
		t.Run(tt.description, func(t *testing.T) {
			// Ensure other Params fields that might be accessed (even if not relevant to LR calc) have defaults
			// if they are not part of the specific test setup for LR.
			// Here, we assume calculateCurrentLr only depends on the LR-specific fields in Params.
			// If it accessed, for example, tt.params.Decay for a "cosine" schedule by mistake,
			// that field should be set to avoid NaN or zero behavior if it's zero-value by default.
			// However, based on calculateCurrentLr logic, only LrSchedule, InitialLr, TargetLr, WarmupSteps, DecaySteps are used.
			// The Decay field is only used by the external training loop for exponential.
			
			actualLr := calculateCurrentLr(&tt.params, tt.currentGlobalStep, tt.totalTrainingSteps)
			if !floatEquals(actualLr, tt.expectedLr, tolerance) {
				t.Errorf("calculateCurrentLr() got = %v, want %v for case '%s'", actualLr, tt.expectedLr, tt.description)
			}
		})
	}
}
