'use strict';

const RLGame = require('./rl-game');

module.exports = function(io) {
    const rlNs = io.of('/rl');
    // Array of RLGame instances for batched environments
    let games = [new RLGame()];

    rlNs.on('connection', (socket) => {
        console.log(`[RL Socket] Connected: ${socket.id}`);

        socket.on('reset', (data, ack) => {
            try {
                const result = games[0].reset();
                if (ack) ack({ status: 'ok', data: result });
            } catch (err) {
                if (ack) ack({ status: 'error', message: err.message });
            }
        });

        socket.on('step', (data, ack) => {
            try {
                const action = data.action || {};
                const result = games[0].step(action);
                if (ack) ack({ status: 'ok', data: result });
            } catch (err) {
                if (ack) ack({ status: 'error', message: err.message });
            }
        });

        socket.on('reset_batch', (data, ack) => {
            try {
                const numAgents = data.num_agents || 1;
                games = [];
                const results = [];
                for(let i = 0; i < numAgents; i++) {
                    const g = new RLGame();
                    games.push(g);
                    results.push(g.reset());
                }
                if (ack) ack({ status: 'ok', data: results });
            } catch (err) {
                if (ack) ack({ status: 'error', message: err.message });
            }
        });

        socket.on('step_batch', (data, ack) => {
            try {
                const actions = data.actions || [];
                if (actions.length !== games.length) {
                    throw new Error(`Expected ${games.length} actions, got ${actions.length}`);
                }
                
                const results = [];
                for(let i = 0; i < games.length; i++) {
                    if (games[i].done) {
                        games[i].reset();
                    }
                    
                    const stepData = games[i].step(actions[i]);
                    
                    if (games[i].done) {
                        const terminalState = stepData.state;
                        const terminalInfo = stepData.info;
                        
                        const resetData = games[i].reset();
                        
                        stepData.state = resetData.state;
                        stepData.info = resetData.info || {};
                        stepData.info.final_state = terminalState;
                        stepData.info.final_info = terminalInfo;
                    }
                    results.push(stepData);
                }
                if (ack) ack({ status: 'ok', data: results });
            } catch (err) {
                if (ack) ack({ status: 'error', message: err.message });
            }
        });

        socket.on('config', (data, ack) => {
            try {
                const config = require('../../config');
                if (ack) ack({
                    status: 'ok',
                    data: {
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
                    }
                });
            } catch (err) {
                if (ack) ack({ status: 'error', message: err.message });
            }
        });

        socket.on('disconnect', () => {
            console.log(`[RL Socket] Disconnected: ${socket.id}`);
        });
    });
};
