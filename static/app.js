// --- State ---
let ws = null;
let gameState = null;

// --- Seat positions (percentage of table area) ---
// Human always at bottom center, AIs distributed around the top
const SEAT_LAYOUTS = {
    2: [
        { top: 105, left: 50 },   // seat 0: human (bottom center)
        { top: -5, left: 50 },    // seat 1: top center
    ],
    3: [
        { top: 105, left: 50 },
        { top: 10, left: 18 },
        { top: 10, left: 82 },
    ],
    4: [
        { top: 105, left: 50 },
        { top: 50, left: 2 },
        { top: -5, left: 50 },
        { top: 50, left: 98 },
    ],
    5: [
        { top: 105, left: 50 },
        { top: 70, left: 2 },
        { top: 5, left: 20 },
        { top: 5, left: 80 },
        { top: 70, left: 98 },
    ],
    6: [
        { top: 105, left: 50 },
        { top: 70, left: 2 },
        { top: 10, left: 12 },
        { top: -5, left: 50 },
        { top: 10, left: 88 },
        { top: 70, left: 98 },
    ],
    7: [
        { top: 105, left: 50 },
        { top: 75, left: 2 },
        { top: 25, left: 2 },
        { top: -5, left: 30 },
        { top: -5, left: 70 },
        { top: 25, left: 98 },
        { top: 75, left: 98 },
    ],
    8: [
        { top: 105, left: 50 },
        { top: 80, left: 2 },
        { top: 35, left: 2 },
        { top: 0, left: 22 },
        { top: -5, left: 50 },
        { top: 0, left: 78 },
        { top: 35, left: 98 },
        { top: 80, left: 98 },
    ],
};

// --- Card Rendering ---
const SUIT_SYMBOLS = { h: '\u2665', d: '\u2666', c: '\u2663', s: '\u2660' };
const SUIT_COLORS = { h: 'red', d: 'red', c: 'black', s: 'black' };
const RANK_DISPLAY = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
    '7': '7', '8': '8', '9': '9', 'T': '10',
    'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
};

function createCardEl(code) {
    if (!code || code === 'back') {
        const el = document.createElement('div');
        el.className = 'card card-back';
        return el;
    }
    const rank = code[0];
    const suit = code[1];
    const el = document.createElement('div');
    const isRed = SUIT_COLORS[suit] === 'red';
    el.className = `card card-front${isRed ? ' red' : ''}`;
    el.innerHTML = `<span class="card-rank">${RANK_DISPLAY[rank]}</span><span class="card-suit">${SUIT_SYMBOLS[suit]}</span>`;
    return el;
}

// --- Chip Layer (casino chip animations + table stack display) ---
let lastHandNumber = null;

const ChipLayer = (() => {
    let betChips = {};          // { playerIdx: [chipEl, ...] }
    let potChips = [];          // chips currently in the pot
    let stackEls = {};          // { playerIdx: containerEl } — persistent stack displays
    let stackNumPlayers = 0;
    let animChips = new Set();  // all animated (non-stack) chips for targeted cleanup
    let containerEl = null;

    function chipColorClass(amount) {
        if (amount >= 500) return 'chip-black';
        if (amount >= 100) return 'chip-green';
        if (amount >= 25)  return 'chip-red';
        return 'chip-white';
    }

    function numAnimChips(amount) {
        if (amount >= 1000) return 4;
        if (amount >= 200)  return 3;
        if (amount >= 50)   return 2;
        return 1;
    }

    function makeAnimChip(colorClass, leftPct, topPct) {
        const el = document.createElement('div');
        el.className = `chip ${colorClass}`;
        el.style.left = `${leftPct}%`;
        el.style.top  = `${topPct}%`;
        containerEl.appendChild(el);
        animChips.add(el);
        return el;
    }

    function betPos(playerIdx, numPlayers) {
        const layout = SEAT_LAYOUTS[numPlayers] || SEAT_LAYOUTS[6];
        const p = layout[playerIdx] || layout[0];
        return {
            left: p.left + (50 - p.left) * 0.4,
            top:  p.top  + (50 - p.top)  * 0.4,
        };
    }

    // Stack position: 25% of the way from player toward center
    function stackPos(playerIdx, numPlayers) {
        const layout = SEAT_LAYOUTS[numPlayers] || SEAT_LAYOUTS[6];
        const p = layout[playerIdx] || layout[0];
        return {
            left: p.left + (50 - p.left) * 0.25,
            top:  p.top  + (50 - p.top)  * 0.25,
        };
    }

    const POT_POS = { left: 50, top: 66 };

    function init() {
        containerEl = document.getElementById('chip-layer');
    }

    // Remove only animated bet/pot chips; preserve stack containers
    function clearAll() {
        animChips.forEach(el => { if (el.parentNode) el.parentNode.removeChild(el); });
        animChips.clear();
        betChips = {};
        potChips = [];
    }

    function clearStacks() {
        Object.values(stackEls).forEach(el => { if (el.parentNode) el.parentNode.removeChild(el); });
        stackEls = {};
        stackNumPlayers = 0;
    }

    // Full reset for new game
    function resetAll() {
        clearAll();
        clearStacks();
    }

    // --- Stack display on table ---
    function stackChipColor(chips) {
        if (chips >= 500) return 'chip-black';
        if (chips >= 100) return 'chip-green';
        if (chips >= 25)  return 'chip-red';
        return 'chip-white';
    }

    function numStackChips(chips) {
        if (chips <= 0)    return 0;
        if (chips >= 1000) return 5;
        if (chips >= 400)  return 4;
        if (chips >= 150)  return 3;
        if (chips >= 50)   return 2;
        return 1;
    }

    function buildStackHTML(chips) {
        if (chips <= 0) {
            return '<div class="chip-stack-label" style="color:#ff6b6b">$0</div>';
        }
        const colorClass = stackChipColor(chips);
        const n = numStackChips(chips);
        let inner = '<div class="chip-stack-chips">';
        for (let i = 0; i < n; i++) inner += `<div class="chip ${colorClass}"></div>`;
        inner += `</div><div class="chip-stack-label">$${chips}</div>`;
        return inner;
    }

    function renderStacks(players) {
        if (!containerEl) return;
        const numPlayers = players.length;
        if (numPlayers !== stackNumPlayers) {
            clearStacks();
            stackNumPlayers = numPlayers;
        }
        players.forEach((p, i) => {
            const pos = stackPos(i, numPlayers);
            if (!stackEls[i]) {
                const el = document.createElement('div');
                el.className = 'chip-stack-container';
                el.style.left = `${pos.left}%`;
                el.style.top  = `${pos.top}%`;
                containerEl.appendChild(el);
                stackEls[i] = el;
            }
            stackEls[i].innerHTML = buildStackHTML(p.chips);
        });
    }

    // --- Bet chip animations ---
    function animateBetChips(playerIdx, amount, numPlayers) {
        if (!containerEl) return;
        const layout = SEAT_LAYOUTS[numPlayers] || SEAT_LAYOUTS[6];
        const src = layout[playerIdx] || layout[0];
        const dst = betPos(playerIdx, numPlayers);
        const colorClass = chipColorClass(amount);
        const count = numAnimChips(amount);
        const chips = [];

        for (let i = 0; i < count; i++) {
            const el = makeAnimChip(colorClass, src.left, src.top);
            chips.push(el);
            const offsetL = (i - (count - 1) / 2) * 0.8;
            const offsetT = -i * 0.5;
            el.classList.add('chip-entering');
            setTimeout(() => {
                el.classList.add('chip-animating');
                el.style.left = `${dst.left + offsetL}%`;
                el.style.top  = `${dst.top  + offsetT}%`;
            }, 30 + i * 40);
        }

        betChips[playerIdx] = (betChips[playerIdx] || []).concat(chips);
    }

    function animateBetsIntoPot(onComplete) {
        if (!containerEl) return;
        const all = Object.values(betChips).flat();
        if (all.length === 0) { if (onComplete) onComplete(); return; }

        all.forEach((el, i) => {
            setTimeout(() => {
                el.classList.add('chip-animating');
                const scatter = (Math.random() - 0.5) * 3;
                el.style.left = `${POT_POS.left + scatter}%`;
                el.style.top  = `${POT_POS.top  + scatter * 0.4}%`;
                el.style.opacity = '0.85';
            }, i * 20);
        });

        const delay = 500 + all.length * 20;
        setTimeout(() => {
            potChips = potChips.concat(all);
            betChips = {};
            if (onComplete) onComplete();
        }, delay);
    }

    function animatePotToWinner(winners, players) {
        if (!containerEl || !winners || winners.length === 0) return;
        if (potChips.length === 0) { clearAll(); return; }

        const numPlayers = players.length;
        const layout = SEAT_LAYOUTS[numPlayers] || SEAT_LAYOUTS[6];
        const winnerIdxs = winners
            .map(w => players.findIndex(p => p.name === w.name))
            .filter(idx => idx !== -1);

        if (winnerIdxs.length === 0) { clearAll(); return; }

        const perWinner = Math.ceil(potChips.length / winnerIdxs.length);
        winnerIdxs.forEach((seatIdx, wi) => {
            const wPos = layout[seatIdx] || layout[0];
            const slice = potChips.slice(wi * perWinner, (wi + 1) * perWinner);
            slice.forEach((el, i) => {
                setTimeout(() => {
                    el.classList.add('chip-animating');
                    el.style.left = `${wPos.left}%`;
                    el.style.top  = `${wPos.top}%`;
                    el.style.opacity = '0';
                }, i * 18);
            });
        });

        setTimeout(clearAll, 700 + potChips.length * 18);
    }

    function update(newState, oldState) {
        if (!containerEl || !newState || !newState.players) return;

        // New hand → clear animated chips then refresh stacks
        if (!oldState || newState.hand_number !== oldState.hand_number) {
            clearAll();
            renderStacks(newState.players);
            return;
        }

        const oldPhase = oldState.phase;
        const newPhase = newState.phase;
        const hadBets  = oldState.players.some(p => p.current_bet > 0);

        // Result: bets → pot → winner
        if (newPhase === 'result' && oldPhase !== 'result') {
            if (hadBets) {
                animateBetsIntoPot(() => animatePotToWinner(newState.winners, newState.players));
            } else {
                animatePotToWinner(newState.winners, newState.players);
            }
            renderStacks(newState.players);
            return;
        }

        // Street change → collect bets to pot
        if (newPhase !== oldPhase && ['preflop','flop','turn','river'].includes(oldPhase)) {
            if (hadBets) animateBetsIntoPot(null);
            renderStacks(newState.players);
            return;
        }

        // Individual bet increases
        newState.players.forEach((p, i) => {
            const oldBet = oldState.players[i] ? (oldState.players[i].current_bet || 0) : 0;
            const delta  = (p.current_bet || 0) - oldBet;
            if (delta > 0 && !p.is_folded) {
                animateBetChips(i, delta, newState.players.length);
            }
        });

        renderStacks(newState.players);
    }

    return { init, update, clearAll, resetAll };
})();

// --- Setup ---
document.addEventListener('DOMContentLoaded', () => {
    ChipLayer.init();
    const setupScreen = document.getElementById('setup-screen');
    const numPlayersSlider = document.getElementById('num-players');
    const numPlayersVal = document.getElementById('num-players-val');
    const startBtn = document.getElementById('start-btn');

    numPlayersSlider.addEventListener('input', () => {
        numPlayersVal.textContent = numPlayersSlider.value;
    });

    startBtn.addEventListener('click', () => {
        const name = document.getElementById('player-name').value.trim() || 'Player';
        const numPlayers = parseInt(numPlayersSlider.value);
        const chips = parseInt(document.getElementById('starting-chips').value) || 1000;
        const blinds = parseInt(document.getElementById('small-blind').value) || 10;

        setupScreen.classList.add('hidden');
        connectAndStart(name, numPlayers, chips, blinds);
    });
});

function connectAndStart(name, numPlayers, chips, smallBlind) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        ws.send(JSON.stringify({
            type: 'new_game',
            name: name,
            num_players: numPlayers,
            starting_chips: chips,
            small_blind: smallBlind,
        }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
            const oldState = gameState;
            gameState = msg;
            ChipLayer.update(gameState, oldState);
            if (msg.gto_suggestions) {
                console.log('[GTO] Received suggestions:', msg.gto_suggestions);
            }
            if (msg.your_turn) {
                console.log('[GTO] Your turn, gto_suggestions:', msg.gto_suggestions ? 'present' : 'null');
            }
            render();
        }
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
    };
}

// --- Rendering ---
function render() {
    if (!gameState) return;

    renderInfoBar();
    renderPlayers();
    renderCommunityCards();
    renderPot();
    renderControls();
    renderAnnouncement();
    renderGameOver();
    renderGTOPanel();
}

function renderInfoBar() {
    const bar = document.getElementById('info-bar');
    const phase = gameState.phase;
    const phaseNames = {
        preflop: 'Pre-Flop', flop: 'Flop', turn: 'Turn',
        river: 'River', showdown: 'Showdown', result: 'Result',
        waiting: 'Waiting'
    };
    bar.textContent = `Hand #${gameState.hand_number} | ${phaseNames[phase] || phase} | Blinds: ${gameState.small_blind}/${gameState.big_blind}`;
}

function renderPlayers() {
    const table = document.getElementById('table');
    // Remove old seats
    table.querySelectorAll('.player-seat').forEach(el => el.remove());

    const players = gameState.players;
    const layout = SEAT_LAYOUTS[players.length] || SEAT_LAYOUTS[6];

    const isNewHand = gameState.hand_number !== lastHandNumber;
    if (isNewHand) lastHandNumber = gameState.hand_number;

    players.forEach((p, i) => {
        const pos = layout[i] || layout[0];
        const seat = document.createElement('div');
        seat.className = 'player-seat';
        if (p.is_human) seat.classList.add('human');
        if (p.is_folded) seat.classList.add('folded');
        if (i === gameState.current_player_index && gameState.phase !== 'result') {
            seat.classList.add('active');
        }

        seat.style.top = `${pos.top}%`;
        seat.style.left = `${pos.left}%`;

        let statusHTML = '';
        if (p.is_folded) statusHTML = '<div class="player-status">Folded</div>';
        else if (p.is_all_in) statusHTML = '<div class="player-status" style="color:#ff9800">ALL IN</div>';

        let betHTML = '';
        if (p.current_bet > 0) {
            betHTML = `<div class="player-bet">Bet: $${p.current_bet}</div>`;
        }

        let badgeHTML = '';
        if (!p.is_human && p.ai_style) {
            badgeHTML = `<span class="ai-badge ${p.ai_style}">${p.ai_style}</span>`;
        }

        let dealerHTML = '';
        if (i === gameState.dealer_index) {
            dealerHTML = '<div class="dealer-chip">D</div>';
        }

        seat.innerHTML = `
            <div class="player-info">
                ${dealerHTML}
                <div class="player-name">${escapeHtml(p.name)} ${badgeHTML}</div>
                <div class="player-chips">$${p.chips}</div>
                ${betHTML}
                ${statusHTML}
            </div>
            <div class="player-cards" id="cards-${i}"></div>
        `;

        // Render cards with optional deal animation
        const cardsDiv = seat.querySelector(`#cards-${i}`);
        if (p.hole_cards) {
            p.hole_cards.forEach((code, cardIdx) => {
                const card = createCardEl(code);
                if (isNewHand) {
                    const n = players.length;
                    const dealOrder = ((i - gameState.dealer_index - 1 + n) % n);
                    card.style.animationDelay = `${(cardIdx * n + dealOrder) * 110}ms`;
                    card.classList.add('card-deal');
                }
                cardsDiv.appendChild(card);
            });
        }

        table.appendChild(seat);
    });
}

function renderCommunityCards() {
    const area = document.getElementById('community-area');
    area.innerHTML = '';

    if (gameState.community_cards) {
        gameState.community_cards.forEach(code => {
            area.appendChild(createCardEl(code));
        });
    }

    // Show empty slots for remaining community cards
    const shown = gameState.community_cards ? gameState.community_cards.length : 0;
    for (let i = shown; i < 5; i++) {
        const placeholder = document.createElement('div');
        placeholder.className = 'card card-back';
        placeholder.style.opacity = '0.2';
        area.appendChild(placeholder);
    }
}

function renderPot() {
    const el = document.getElementById('pot-display');
    el.textContent = `Pot: $${gameState.pot}`;
}

function renderControls() {
    const controls = document.getElementById('controls');

    if (!gameState.your_turn || gameState.game_over) {
        controls.classList.add('hidden');
        return;
    }

    controls.classList.remove('hidden');
    const va = gameState.valid_actions;

    const foldBtn = document.getElementById('fold-btn');
    const checkBtn = document.getElementById('check-btn');
    const callBtn = document.getElementById('call-btn');
    const raiseBtn = document.getElementById('raise-btn');
    const raiseSlider = document.getElementById('raise-amount');
    const raiseValue = document.getElementById('raise-value');
    const raiseArea = document.getElementById('raise-slider-area');

    foldBtn.disabled = false;

    if (va.can_check) {
        checkBtn.style.display = '';
        checkBtn.disabled = false;
        callBtn.style.display = 'none';
    } else {
        checkBtn.style.display = 'none';
        callBtn.style.display = '';
        callBtn.disabled = !va.can_call;
        callBtn.textContent = `Call $${va.call_amount}`;
    }

    if (va.can_raise) {
        raiseBtn.disabled = false;
        raiseArea.style.display = '';
        raiseSlider.min = va.min_raise;
        raiseSlider.max = va.max_raise;
        raiseSlider.value = va.min_raise;
        raiseValue.textContent = `$${va.min_raise}`;

        raiseSlider.oninput = () => {
            raiseValue.textContent = `$${raiseSlider.value}`;
        };
    } else {
        raiseBtn.disabled = true;
        raiseArea.style.display = 'none';
    }
}

function renderAnnouncement() {
    const el = document.getElementById('announcement');

    if (gameState.phase === 'result' && gameState.winners) {
        el.classList.remove('hidden');
        let html = '<h2>Hand Result</h2>';
        gameState.winners.forEach(w => {
            html += `<p><strong>${escapeHtml(w.name)}</strong> wins $${w.chips_won}</p>`;
            html += `<p class="hand-name">${escapeHtml(w.hand_name)}</p>`;
        });
        el.innerHTML = html;
    } else {
        el.classList.add('hidden');
    }
}

function renderGameOver() {
    const el = document.getElementById('game-over');
    if (gameState.game_over) {
        el.classList.remove('hidden');
        const human = gameState.players.find(p => p.is_human);
        const isWinner = human && human.chips > 0;
        el.querySelector('h1').textContent = isWinner ? 'You Win!' : 'Game Over';
        el.querySelector('p').textContent = isWinner
            ? `Final chips: $${human.chips}`
            : 'You ran out of chips!';
    } else {
        el.classList.add('hidden');
    }
}

// --- GTO Panel ---
const ACTION_BAR_CLASSES = {
    'fold': 'gto-bar-fold',
    'check/call': 'gto-bar-check-call',
    'bet half pot': 'gto-bar-bet-half',
    'bet pot': 'gto-bar-bet-pot',
    'all-in': 'gto-bar-all-in',
};

const ACTION_ORDER = ['fold', 'check/call', 'bet half pot', 'bet pot', 'all-in'];

const BUCKET_LABELS = [
    'Trash', 'Weak', 'Below Avg', 'Marginal', 'Playable',
    'Decent', 'Good', 'Strong', 'Premium', 'Monster'
];

function getBucketColor(bucket) {
    const colors = [
        '#c62828', '#d84315', '#e65100', '#f57f17', '#9e9d24',
        '#558b2f', '#2e7d32', '#00695c', '#00838f', '#1565c0'
    ];
    return colors[bucket] || '#555';
}

function getEquityColor(equity) {
    if (equity >= 70) return '#4caf50';
    if (equity >= 50) return '#8bc34a';
    if (equity >= 35) return '#ffc107';
    if (equity >= 20) return '#ff9800';
    return '#f44336';
}

function renderGTOPanel() {
    const panel = document.getElementById('gto-panel');
    const content = document.getElementById('gto-content');
    const gto = gameState.gto_suggestions;

    if (!gto || !gameState.your_turn) {
        panel.classList.add('hidden');
        document.getElementById('game-area').style.marginRight = '0';
        return;
    }

    panel.classList.remove('hidden');
    panel.style.display = 'flex';
    document.getElementById('game-area').style.marginRight = '320px';

    let html = '';

    // Algorithm info badge
    if (gto.algorithm) {
        const algo = gto.algorithm;
        html += `<div class="gto-algo-badge">`;
        html += `<span class="gto-algo-name">${algo.name}</span>`;
        html += `<span class="gto-algo-detail">${(algo.iterations / 1000).toFixed(0)}k iters</span>`;
        html += `<span class="gto-algo-sep">&middot;</span>`;
        html += `<span class="gto-algo-detail">${algo.abstraction}</span>`;
        html += `</div>`;
    }

    // Hand Strength section with equity gauge
    const eqColor = getEquityColor(gto.equity);
    html += `<div class="gto-section">`;
    html += `<div class="gto-section-title">Hand Strength <span class="gto-phase-badge">${gto.phase}</span></div>`;
    html += `<div class="gto-hand-strength">`;
    html += `<div class="gto-equity-ring" style="--eq-pct:${gto.equity}; --eq-color:${eqColor}">`;
    html += `<div class="gto-equity-inner"><span class="gto-equity-num">${gto.equity}%</span><span class="gto-equity-sub">equity</span></div>`;
    html += `</div>`;
    html += `<div class="gto-strength-info">`;
    html += `<div class="gto-bucket-badge" style="background:${getBucketColor(gto.bucket)}">${BUCKET_LABELS[gto.bucket] || 'Bucket ' + gto.bucket}</div>`;
    html += `<div class="gto-bucket-num">Strength ${gto.bucket}/9</div>`;
    html += `</div>`;
    html += `</div></div>`;

    // Recommended action callout with GTO Play button
    if (gto.recommended) {
        const recPct = gto.your_strategy[gto.recommended];
        const recClass = ACTION_BAR_CLASSES[gto.recommended] || 'gto-bar-check-call';
        html += `<div class="gto-recommendation">`;
        html += `<div class="gto-rec-label">Recommended</div>`;
        html += `<div class="gto-rec-action ${recClass}">${gto.recommended}</div>`;
        if (recPct !== undefined) {
            html += `<div class="gto-rec-freq">${recPct}% frequency</div>`;
        }
        html += `</div>`;
    }

    // Strategy distribution
    html += `<div class="gto-section">`;
    html += `<div class="gto-section-title">Strategy Distribution</div>`;
    for (const action of ACTION_ORDER) {
        const pct = gto.your_strategy[action];
        if (pct === undefined) continue;
        const barClass = ACTION_BAR_CLASSES[action] || 'gto-bar-check-call';
        const isRec = gto.recommended === action;
        html += `<div class="gto-action-row${isRec ? ' gto-action-recommended' : ''}">`;
        html += `<div class="gto-action-name">${action}${isRec ? ' ★' : ''}</div>`;
        html += `<div class="gto-action-bar-bg"><div class="gto-action-bar-fill ${barClass}" style="width:${pct}%"></div></div>`;
        html += `<div class="gto-action-pct">${pct}%</div>`;
        html += `</div>`;
    }
    html += `</div>`;

    // Full bucket table
    html += `<div class="gto-section">`;
    html += `<div class="gto-section-title">All Buckets &mdash; ${gto.phase}</div>`;
    html += `<table class="gto-bucket-table"><thead><tr>`;
    html += `<th>Str</th><th>Fold</th><th>Chk/Call</th><th>&frac12; Pot</th><th>Pot</th><th>All-in</th>`;
    html += `</tr></thead><tbody>`;

    for (const b of gto.all_buckets) {
        const isActive = b.bucket === gto.bucket;
        html += `<tr class="${isActive ? 'active-bucket' : ''}">`;
        html += `<td class="bucket-label" style="color:${getBucketColor(b.bucket)}" title="${BUCKET_LABELS[b.bucket]}">${b.bucket}</td>`;
        for (const action of ACTION_ORDER) {
            const val = b.strategy[action];
            if (val !== undefined) {
                const intensity = Math.min(val / 100, 1);
                const bg = `rgba(255,255,255,${(intensity * 0.12).toFixed(2)})`;
                html += `<td style="background:${bg}">${val}%</td>`;
            } else {
                html += `<td>-</td>`;
            }
        }
        html += `</tr>`;
    }

    html += `</tbody></table></div>`;

    // GTO Play suggestion button
    html += `<div class="gto-section">`;
    html += `<button class="gto-play-btn" onclick="gtoRoll()">Action Suggestion</button>`;
    html += `<div id="gto-roll-result"></div>`;
    html += `</div>`;

    content.innerHTML = html;
}

// --- Actions ---
function sendAction(action, amount) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'action', action, amount: parseInt(amount) || 0 }));
    document.getElementById('controls').classList.add('hidden');
}

function doFold() { sendAction('fold'); }
function doCheck() { sendAction('check'); }
function doCall() { sendAction('call'); }
function doRaise() {
    const amount = parseInt(document.getElementById('raise-amount').value);
    sendAction('raise', amount);
}

function gtoRoll() {
    if (!gameState || !gameState.gto_suggestions) return;
    const gto = gameState.gto_suggestions;

    // Build weighted action list from strategy distribution
    const entries = [];
    for (const action of ACTION_ORDER) {
        const pct = gto.your_strategy[action];
        if (pct !== undefined && pct > 0) {
            entries.push({ action, weight: pct });
        }
    }
    if (entries.length === 0) return;

    // Weighted random selection
    const totalWeight = entries.reduce((s, e) => s + e.weight, 0);
    let roll = Math.random() * totalWeight;
    let chosen = entries[entries.length - 1].action;
    for (const e of entries) {
        roll -= e.weight;
        if (roll <= 0) { chosen = e.action; break; }
    }

    const chosenPct = gto.your_strategy[chosen] || 0;
    const barClass = ACTION_BAR_CLASSES[chosen] || 'gto-bar-check-call';

    const el = document.getElementById('gto-roll-result');
    el.innerHTML = `<div class="gto-roll-suggestion">` +
        `<div class="gto-roll-label">You should:</div>` +
        `<div class="gto-roll-action ${barClass}">${chosen}</div>` +
        `<div class="gto-roll-freq">(${chosenPct}% of the time)</div>` +
        `</div>`;
}

function playAgain() {
    ChipLayer.resetAll();
    document.getElementById('game-over').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden');
    if (ws) ws.close();
}

// --- Util ---
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
