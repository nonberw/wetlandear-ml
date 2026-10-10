// Runs the app's voice-worker training code in Node on a real case (5 recordings of one species vs shipped negatives).
const fs = require('fs'), path = require('path');
const T = process.env.TMP_DIR || 'C:/Users/Timur/AppData/Local/Temp';
function npy(file) {   // minimal .npy reader for float32 2-D arrays
  const b = fs.readFileSync(file), hl = b.readUInt16LE(8), hdr = b.slice(10, 10 + hl).toString(), shape = /\((\d+), (\d+)\)/.exec(hdr), n = +shape[1], d = +shape[2];
  const data = new Float32Array(b.buffer.slice(b.byteOffset + 10 + hl, b.byteOffset + b.length)); return { data, n, d };
}
const src = fs.readFileSync('C:/Users/Timur/Documents/wetlandear-app/js/voice-worker.js', 'utf8');
global.self = { postMessage() {} };
eval(src + '\n;global.train = train; global.score = score;');
const tr = npy(T + '/vt_train.npy'), te = npy(T + '/vt_test.npy'), nt = npy(T + '/vt_neg.npy');
const negBin = fs.readFileSync('C:/Users/Timur/Documents/wetlandear-app/models/negatives.bin'), meta = JSON.parse(fs.readFileSync('C:/Users/Timur/Documents/wetlandear-app/models/negatives.json', 'utf8'));
const N = new Float32Array(negBin.length); for (let i = 0; i < negBin.length; i++) N[i] = (negBin[i] > 127 ? negBin[i] - 256 : negBin[i]) / 127;
// the species is Fulica atra: leave its rows out of the shipped negatives
const keep = []; for (let i = 0; i < meta.n; i++) if (meta.sci[i] !== 'Fulica atra') keep.push(i);
const N2 = new Float32Array(keep.length * meta.d); keep.forEach((r, k) => N2.set(N.subarray(r * meta.d, (r + 1) * meta.d), k * meta.d));
const t0 = Date.now();
const m = train(tr.data, tr.n, N2, keep.length, tr.d, 300, 2.0, 1e-3);
const ms = Date.now() - t0;
const sp = score(m, te.data, te.n, te.d), sn = score(m, nt.data, nt.n, nt.d);
const rate = (a, t) => (Array.from(a).filter(x => x >= t).length / a.length * 100).toFixed(1) + '%';
console.log('train ms', ms, '| positives', te.n, 'negatives', nt.n);
for (const t of [0.3, 0.4, 0.5, 0.7]) console.log('p>=' + t, 'found', rate(sp, t), 'false alarms', rate(sn, t));
