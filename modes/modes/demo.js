module.exports = { onStart: async () => { console.log('Demo mode: no real trades'); let price = await getLivePrice('BTC'); return { price }; }, payWall: true, };
