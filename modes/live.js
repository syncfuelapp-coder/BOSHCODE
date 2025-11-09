module.exports = { onStart: async () => { console.log('Live trading ON-checking market...'); let price = await getLivePrice('BTC'); return { price }; }, payWall: false, };
