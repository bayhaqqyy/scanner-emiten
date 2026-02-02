const fmt = (value) => {
  if (value === null || value === undefined) return "-";
  return value;
};

const statusBadge = (status) => {
  const label = status === "signal" ? "SIGNAL" : "WATCH";
  const cls = status === "signal" ? "badge badge-signal" : "badge badge-watch";
  return `<span class="${cls}">${label}</span>`;
};

const buildTable = (items) => {
  if (!items || items.length === 0) {
    return '<div class="empty">Belum ada sinyal sesuai rule. Coba beberapa saat lagi.</div>';
  }
  const header = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Score</th>
            <th>Ticker</th>
            <th>Close</th>
            <th>Change%</th>
            <th>RSI</th>
            <th>Vol</th>
            <th>Entry</th>
            <th>SL</th>
            <th>TP</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr>
        <td>${statusBadge(item.status)}</td>
        <td>${fmt(item.score)}</td>
        <td>${fmt(item.ticker)}</td>
        <td>${fmt(item.close)}</td>
        <td>${fmt(item.change_pct)}</td>
        <td>${fmt(item.rsi)}</td>
        <td>${fmt(item.vol_spike)}</td>
        <td>${fmt(item.entry)}</td>
        <td>${fmt(item.sl)}</td>
        <td>${fmt(item.tp)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
};

async function refresh() {
  const [scalping, swing] = await Promise.all([
    fetch("/api/scalping").then((r) => r.json()),
    fetch("/api/swing").then((r) => r.json()),
  ]);

  document.getElementById("scalping-updated").textContent = scalping.updated_at || "-";
  document.getElementById("swing-updated").textContent = swing.updated_at || "-";
  document.getElementById("scalping-error").textContent = scalping.error || "";
  document.getElementById("swing-error").textContent = swing.error || "";
  document.getElementById("scalping-table").innerHTML = buildTable(scalping.items);
  document.getElementById("swing-table").innerHTML = buildTable(swing.items);
}

refresh();
setInterval(refresh, window.UI_POLL_SECONDS ? window.UI_POLL_SECONDS * 1000 : 1000);
