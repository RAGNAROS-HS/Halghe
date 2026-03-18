/*
 * rl-api.js — Express router exposing the RL training API.
 *
 * POST /rl/reset   → reset episode, return initial state
 * POST /rl/step    → apply action, tick once, return {state, reward, done, info}
 * GET  /rl/config  → return game configuration
 */

'use strict';

const express = require('express');
const router = express.Router();
const RLGame = require('./rl-game');

// Single RL game instance (one episode at a time).
// For multi-agent training in the future, this could become a Map<sessionId, RLGame>.
let game = new RLGame();

/**
 * POST /rl/reset
 * Body: (none required)
 * Response: { state, info }
 */
router.post('/reset', (req, res) => {
    try {
        const result = game.reset();
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
        const result = game.step(action);
        res.json(result);
    } catch (err) {
        res.status(400).json({ error: err.message });
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
