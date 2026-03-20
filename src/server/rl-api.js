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

// Array of RLGame instances for batched environments
let games = [new RLGame()];

/**
 * POST /rl/reset
 * Body: (none required)
 * Response: { state, info }
 */
router.post('/reset', (req, res) => {
    try {
        // Fallback for single-agent API: reset the first game
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
 * Response: Array of { state, info }
 */
router.post('/reset_batch', (req, res) => {
    try {
        const numAgents = req.body.num_agents || 1;
        games = [];
        const results = [];
        for(let i = 0; i < numAgents; i++) {
            const g = new RLGame();
            games.push(g);
            results.push(g.reset());
        }
        res.json(results);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * POST /rl/step_batch
 * Body: { actions: [ {dx, dy, split, fire}, ... ] }
 * Response: Array of { state, reward, done, info }
 */
router.post('/step_batch', (req, res) => {
    try {
        const actions = req.body.actions || [];
        if (actions.length !== games.length) {
            throw new Error(`Expected ${games.length} actions, got ${actions.length}`);
        }
        
        const results = [];
        for(let i = 0; i < games.length; i++) {
            if (games[i].done) {
                // Should not happen if auto-reset works, but just in case
                games[i].reset();
            }
            
            const stepData = games[i].step(actions[i]);
            
            // Gymnasium VectorEnv standard auto-reset
            if (games[i].done) {
                const terminalState = stepData.state;
                const terminalInfo = stepData.info;
                
                const resetData = games[i].reset();
                
                // Return the new pristine state, but keep the current step's reward and done flag!
                stepData.state = resetData.state;
                stepData.info = resetData.info || {};
                
                // Stash the terminal state in info
                stepData.info.final_state = terminalState;
                stepData.info.final_info = terminalInfo;
            }
            results.push(stepData);
        }
        res.json(results);
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
