module.exports = { onStart: async () => { console.log(Pay mode: charging $0.01 per query); let price = await getLivePrice('BTC'); return { price }; }, payWall: true, pricePerCall: 0.01 };
