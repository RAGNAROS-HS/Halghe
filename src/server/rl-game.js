/*
 * rl-game.js — Self-contained RL training episode.
 *
 * Creates an isolated Map instance (separate from the multiplayer game)
 * with a single bot player. Exposes reset() and step(action) methods
 * that drive the game synchronously — one tick per call.
 *
 * Returns RAW game state as JSON so the Python side can build
 * whatever observation tensor it wants.
 */

'use strict';

const config = require('../../config');
const mapUtils = require('./map/map');
const util = require('./lib/util');
const gameLogic = require('./game-logic');
const { getPosition } = require('./lib/entityUtils');

const INIT_MASS_LOG = util.mathLog(config.defaultPlayerMass, config.slowBase);
const MAX_STEPS_PER_EPISODE = 3000;

class RLGame {
    constructor() {
        this.map = null;
        this.player = null;
        this.stepCount = 0;
        this.prevMass = 0;
        this.done = true;
    }

    /**
     * Reset the episode: fresh map, fresh player.
     * @returns {{ state: object, info: object }}
     */
    reset() {
        this.map = new mapUtils.Map(config);
        this.stepCount = 0;
        this.done = false;

        // Populate food and viruses so there's something to eat
        this.map.balanceMass(config.foodMass, config.gameMass, config.maxFood, config.maxVirus);

        // Create the bot player
        const playerUtils = mapUtils.playerUtils;
        this.player = new playerUtils.Player('rl-bot');
        const radius = util.massToRadius(config.defaultPlayerMass);
        const spawnPos = getPosition(
            config.newPlayerInitialPosition === 'farthest',
            radius,
            this.map.players.data
        );
        this.player.init(spawnPos, config.defaultPlayerMass);
        this.player.name = 'RL_Agent';
        this.player.screenWidth = 1920;
        this.player.screenHeight = 1080;
        this.map.players.pushNew(this.player);

        this.prevMass = this.player.massTotal;

        return {
            state: this._getState(),
            info: {
                step: this.stepCount,
                mass: this.player.massTotal,
            }
        };
    }

    /**
     * Apply an action and tick the game once.
     * @param {{ dx: number, dy: number, split: number, fire: number }} action
     * @returns {{ state: object, reward: number, done: boolean, info: object }}
     */
    step(action) {
        if (this.done) {
            throw new Error('Episode is done. Call reset() first.');
        }

        // --- Apply action ---
        const dx = Math.max(-1, Math.min(1, action.dx || 0));
        const dy = Math.max(-1, Math.min(1, action.dy || 0));

        // Convert [-1,1] direction into a target point offset from the player center.
        // The game's movement system steers towards (player.x + target.x, player.y + target.y),
        // so we set target to a fixed distance in the desired direction.
        const MOVE_DISTANCE = 400;
        this.player.target = {
            x: dx * MOVE_DISTANCE,
            y: dy * MOVE_DISTANCE
        };

        // Split
        if (action.split) {
            this.player.userSplit(config.limitSplit, config.defaultPlayerMass);
        }

        // Fire mass
        if (action.fire) {
            const minCellMass = config.defaultPlayerMass + config.fireFood;
            for (let i = 0; i < this.player.cells.length; i++) {
                if (this.player.cells[i].mass >= minCellMass) {
                    this.player.changeCellMass(i, -config.fireFood);
                    this.map.massFood.addNew(this.player, i, config.fireFood);
                }
            }
        }

        // --- Tick the game ---
        this._tickPlayer();
        this._tickGame();

        // Rebalance food/viruses (normally runs every 1s = every 60 ticks)
        if (this.stepCount % 60 === 0) {
            this.map.players.shrinkCells(config.massLossRate, config.defaultPlayerMass, config.minMassLoss);
            this.map.balanceMass(config.foodMass, config.gameMass, config.maxFood, config.maxVirus);
        }

        this.stepCount++;

        // --- Compute reward ---
        const currentMass = this.player.massTotal;
        let reward = currentMass - this.prevMass;

        // Check if the bot player died (removed from player list, or no cells left)
        const botIndex = this.map.players.findIndexByID('rl-bot');
        if (botIndex === -1 || this.player.cells.length === 0) {
            this.done = true;
            reward -= 10;
        }

        // Episode length limit
        if (this.stepCount >= MAX_STEPS_PER_EPISODE) {
            this.done = true;
        }

        this.prevMass = this.done ? 0 : currentMass;

        return {
            state: this.done ? this._getEmptyState() : this._getState(),
            reward,
            done: this.done,
            info: {
                step: this.stepCount,
                mass: this.done ? 0 : currentMass,
            }
        };
    }

    // ------------------------------------------------------------------
    // Private: game tick logic (mirrors server.js tickPlayer + tickGame)
    // ------------------------------------------------------------------

    _tickPlayer() {
        const currentPlayer = this.player;

        currentPlayer.move(config.slowBase, config.gameWidth, config.gameHeight, INIT_MASS_LOG);

        const isEntityInsideCircle = (point, circleX, circleY, circleRadius) => {
            const dx = point.x - circleX;
            const dy = point.y - circleY;
            return (dx * dx + dy * dy) <= (circleRadius * circleRadius);
        };

        const canEatMass = (cell, cellX, cellY, cellRadius, cellIndex, mass) => {
            if (isEntityInsideCircle(mass, cellX, cellY, cellRadius)) {
                if (mass.id === currentPlayer.id && mass.speed > 0 && cellIndex === mass.num)
                    return false;
                if (cell.mass > mass.mass * 1.1)
                    return true;
            }
            return false;
        };

        const canEatVirus = (cell, cellX, cellY, cellRadius, virus) => {
            return virus.mass < cell.mass && isEntityInsideCircle(virus, cellX, cellY, cellRadius);
        };

        const cellsToSplit = [];
        for (let cellIndex = 0; cellIndex < currentPlayer.cells.length; cellIndex++) {
            const currentCell = currentPlayer.cells[cellIndex];

            const eatenFoodIndexes = util.getIndexes(this.map.food.data, food =>
                isEntityInsideCircle(food, currentCell.x, currentCell.y, currentCell.radius)
            );
            const eatenMassIndexes = util.getIndexes(this.map.massFood.data, mass =>
                canEatMass(currentCell, currentCell.x, currentCell.y, currentCell.radius, cellIndex, mass)
            );
            const eatenVirusIndexes = util.getIndexes(this.map.viruses.data, virus =>
                canEatVirus(currentCell, currentCell.x, currentCell.y, currentCell.radius, virus)
            );

            if (eatenVirusIndexes.length > 0) {
                cellsToSplit.push(cellIndex);
                this.map.viruses.delete(eatenVirusIndexes);
            }

            let massGained = eatenMassIndexes.reduce((acc, index) => acc + this.map.massFood.data[index].mass, 0);
            this.map.food.delete(eatenFoodIndexes);
            this.map.massFood.remove(eatenMassIndexes);
            massGained += (eatenFoodIndexes.length * config.foodMass);
            currentPlayer.changeCellMass(cellIndex, massGained);
        }
        currentPlayer.virusSplit(cellsToSplit, config.limitSplit, config.defaultPlayerMass);
    }

    _tickGame() {
        this.map.massFood.move(config.gameWidth, config.gameHeight);

        // Handle player-vs-player collisions (in case there are bot enemies later)
        this.map.players.handleCollisions((gotEaten, eater) => {
            const cellGotEaten = this.map.players.getCell(gotEaten.playerIndex, gotEaten.cellIndex);
            this.map.players.data[eater.playerIndex].changeCellMass(eater.cellIndex, cellGotEaten.mass);
            const playerDied = this.map.players.removeCell(gotEaten.playerIndex, gotEaten.cellIndex);
            if (playerDied) {
                this.map.players.removePlayerByIndex(gotEaten.playerIndex);
            }
        });
    }

    // ------------------------------------------------------------------
    // Private: state serialization — raw JSON for the Python side
    // ------------------------------------------------------------------

    _getState() {
        const p = this.player;
        return {
            player: {
                x: p.x,
                y: p.y,
                massTotal: Math.round(p.massTotal),
                cells: p.cells.map(c => ({ x: c.x, y: c.y, mass: c.mass, radius: c.radius }))
            },
            food: this.map.food.data.map(f => ({ x: f.x, y: f.y })),
            viruses: this.map.viruses.data.map(v => ({ x: v.x, y: v.y, mass: v.mass, radius: v.radius })),
            enemies: this.map.players.data
                .filter(pl => pl.id !== 'rl-bot')
                .map(pl => ({
                    x: pl.x,
                    y: pl.y,
                    massTotal: Math.round(pl.massTotal),
                    cells: pl.cells.map(c => ({ x: c.x, y: c.y, mass: c.mass, radius: c.radius }))
                })),
            massFood: this.map.massFood.data.map(m => ({ x: m.x, y: m.y, mass: m.mass })),
            map: {
                width: config.gameWidth,
                height: config.gameHeight
            }
        };
    }

    _getEmptyState() {
        return {
            player: { x: 0, y: 0, massTotal: 0, cells: [] },
            food: [],
            viruses: [],
            enemies: [],
            massFood: [],
            map: { width: config.gameWidth, height: config.gameHeight }
        };
    }

    // ------------------------------------------------------------------
    // Efficient observation and render helpers
    // ------------------------------------------------------------------

    /**
     * Returns the 5-element observation vector, computed directly on the server
     * to avoid transmitting the full game state over HTTP.
     * [px, py, pmass, num_food, num_enemies] — all normalized.
     */
    getObservation() {
        if (this.done || !this.player || this.player.cells.length === 0) {
            return [0.0, 0.0, 0.0, 0.0, 0.0];
        }
        const c = this.player.cells[0];
        return [
            c.x / config.gameWidth,
            c.y / config.gameHeight,
            c.mass / 100.0,
            this.map.food.data.length / 100.0,
            0.0  // no enemies in current RL setup
        ];
    }

    /**
     * Returns compact cell data for all of this agent's cells.
     * Used only for rendering — do not call during normal training steps.
     */
    getPlayerCells() {
        if (!this.player || this.player.cells.length === 0) return [];
        return this.player.cells.map(c => ({ x: c.x, y: c.y, radius: c.radius }));
    }

    /**
     * Returns background render data (food, viruses, massFood) for one map instance.
     * Only needed for visualization — do not call during normal training steps.
     */
    getRenderBackground() {
        return {
            food: this.map.food.data.map(f => ({ x: f.x, y: f.y })),
            viruses: this.map.viruses.data.map(v => ({ x: v.x, y: v.y, radius: v.radius })),
            massFood: this.map.massFood.data.map(m => ({ x: m.x, y: m.y })),
            map: { width: config.gameWidth, height: config.gameHeight }
        };
    }
}

module.exports = RLGame;
