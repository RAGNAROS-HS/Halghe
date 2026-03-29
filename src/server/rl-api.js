/*
 * rl-api.js — Express router exposing the RL training API.
 *
 * POST /rl/reset        → reset episode, return initial state (single agent)
 * POST /rl/step         → apply action, tick once, return {state, reward, done, info}
 * POST /rl/reset_batch  → reset N episodes, return slim {obs, info} per agent
 * POST /rl/step_batch   → step N agents, return slim {obs, reward, done, info} per agent
 * GET  /rl/render_state → on-demand full state for visualization (agent 0 + all cells)
 * GET  /rl/config       → return game configuration
 *
 * Slim format: observations are pre-computed on the server to avoid transmitting
 * full game state (food arrays, virus arrays, etc.) over HTTP every step.
 * Full state serialization only happens on the /render_state endpoint.
 */

'use strict';

const express = require('express');
const router = express.Router();
const RLGame = require('./rl-game');

let games = [new RLGame()];

/**
 * POST /rl/reset
 * Body: (none required)
 * Response: { state, info }
 */
router.post('/reset', (req, res) => {
    try {
        const result = games[0].reset();
        res.json(result);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * POST /rl/step
 * Body: { action: { dx, dy, split, fire } }
 * Response: { state, reward, done, info }
 */
router.post('/step', (req, res) => {
    try {
        const action = req.body.action || {};
        const result = games[0].step(action);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

/**
 * POST /rl/reset_batch
 * Body: { num_agents: number }
 * Response: Array of { obs: [px, py, pmass, num_food, num_enemies], info }
 */
router.post('/reset_batch', (req, res) => {
    try {
        const numAgents = req.body.num_agents || 1;
        games = [];
        const results = [];
        for (let i = 0; i < numAgents; i++) {
            const g = new RLGame();
            games.push(g);
            g.reset();
            results.push({
                obs: g.getObservation(),
                info: { step: 0, mass: g.player ? Math.round(g.player.massTotal) : 0 }
            });
        }
        res.json(results);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * POST /rl/step_batch
 * Body: { actions: [ [dx, dy, split, fire], ... ], skip: number }
 *   actions can be either array format [dx, dy, split, fire]
 *   or object format {dx, dy, split, fire} — both are accepted.
 * Response: Array of { obs, reward, done, info }
 *
 * Auto-resets done agents (Gymnasium VectorEnv convention):
 *   returned obs is the INITIAL observation of the new episode,
 *   done=true signals the previous episode ended.
 */
router.post('/step_batch', (req, res) => {
    try {
        const actions = req.body.actions || [];
        const skip = Math.max(1, req.body.skip || 1);
        if (actions.length !== games.length) {
            throw new Error(`Expected ${games.length} actions, got ${actions.length}`);
        }

        const results = [];
        for (let i = 0; i < games.length; i++) {
            const game = games[i];
            if (game.done) {
                game.reset();
            }

            // Accept both array [dx, dy, split, fire] and object {dx, dy, split, fire}
            const raw = actions[i];
            const act = Array.isArray(raw) ? {
                dx: raw[0],
                dy: raw[1],
                split: raw[2] > 0 ? 1 : 0,
                fire: raw[3] > 0 ? 1 : 0
            } : raw;

            let totalReward = 0;
            let done = false;
            for (let s = 0; s < skip; s++) {
                const stepData = game.step(act);
                totalReward += stepData.reward;
                done = stepData.done;
                if (done) break;
            }

            const info = {};
            if (done) {
                // Stash terminal obs before reset (zeros since agent is dead)
                info.final_obs = game.getObservation();
                game.reset();  // auto-reset: next obs is the fresh episode start
            }

            results.push({
                obs: game.getObservation(),
                reward: totalReward,
                done,
                info
            });
        }
        res.json(results);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

/**
 * GET /rl/render_state
 * Returns full visualization state on demand — only called during video recording.
 * Response: { render_bg: { food, viruses, massFood, map }, player_cells: [ [...cells], ... ] }
 */
router.get('/render_state', (req, res) => {
    try {
        if (games.length === 0) {
            return res.json({ render_bg: null, player_cells: [] });
        }
        res.json({
            render_bg: games[0].getRenderBackground(),
            player_cells: games.map(g => g.getPlayerCells())
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * GET /rl/config
 * Response: game configuration (map size, masses, etc.)
 */
router.get('/config', (req, res) => {
    const config = require('../../config');
    res.json({
        gameWidth: config.gameWidth,
        gameHeight: config.gameHeight,
        defaultPlayerMass: config.defaultPlayerMass,
        foodMass: config.foodMass,
        fireFood: config.fireFood,
        limitSplit: config.limitSplit,
        maxFood: config.maxFood,
        maxVirus: config.maxVirus,
        slowBase: config.slowBase,
        massLossRate: config.massLossRate,
        minMassLoss: config.minMassLoss,
    });
});

module.exports = router;
