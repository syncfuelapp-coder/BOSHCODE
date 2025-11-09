module.exports = { onStart: async (days) => { console.log('Backtesting over ' + days + ' days...'); let historical = await getHistory('BTC', days); return { days, historical }; }, payWall: false, };
