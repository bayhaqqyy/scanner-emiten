const fmt = (value) => {
  if (value === null || value === undefined) return "-";
  return value;
};

const statusBadge = (status) => {
  const label = status === "signal" ? "SIGNAL" : "WATCH";
  const cls = status === "signal" ? "badge badge-signal" : "badge badge-watch";
  return `<span class="${cls}">${label}</span>`;
};

const aiBadge = (ai) => {
  if (!ai || !ai.bias) return "-";
  const conf = ai.confidence ? ` ${ai.confidence}/10` : "";
  return `<span class="ai-pill">${ai.bias}${conf}</span>`;
};

const buildTable = (items, kind) => {
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
            <th>AI</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr data-ticker="${fmt(item.ticker)}" data-kind="${kind}">
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
        <td>${aiBadge(item.ai)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
};

const buildCorporateTable = (items) => {
  if (!items || items.length === 0) {
    return '<div class="empty">Belum ada berita corporate action.</div>';
  }
  const header = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Tag</th>
            <th>Perihal</th>
            <th>Tanggal</th>
            <th>Sumber</th>
          </tr>
        </thead>
        <tbody>
  `;
  const rows = items.map((item) => `
      <tr>
        <td><span class="badge badge-watch">${fmt(item.tag)}</span></td>
        <td>${item.url ? `<a href="${item.url}" target="_blank" rel="noreferrer">${fmt(item.title)}</a>` : fmt(item.title)}</td>
        <td>${fmt(item.date)}</td>
        <td>${fmt(item.source)}</td>
      </tr>
  `).join("");
  return header + rows + "</tbody></table></div>";
};

const renderFundamentals = (data) => {
  const ratios = data.ratios || {};
  const inputs = data.inputs || {};
  document.getElementById("fundamental-selected").textContent = data.ticker || "-";
  document.getElementById("fundamental-asof").textContent = data.as_of || "-";

  const rows = [
    ["Market Cap", ratios.market_cap],
    ["PBV", ratios.pbv],
    ["BVPS", ratios.bvps],
    ["PER", ratios.per],
    ["EPS", ratios.eps],
    ["Net Profit Margin", ratios.net_profit_margin],
    ["DER", ratios.der],
    ["ROE", ratios.roe],
    ["Nilai Wajar", ratios.fair_value],
    ["Target Market Cap", ratios.target_market_cap],
    ["PBV Band", ratios.pbv_band],
  ]
    .map(([label, value]) => `
      <div class="fund-row">
        <span>${label}</span>
        <span>${fmt(value)}</span>
      </div>
    `)
    .join("");

  const inputsHtml = `
    <div class="fund-subtitle">Input Data</div>
    <div class="fund-row"><span>Price</span><span>${fmt(inputs.price)}</span></div>
    <div class="fund-row"><span>Shares Outstanding</span><span>${fmt(inputs.shares_outstanding)}</span></div>
    <div class="fund-row"><span>Net Income</span><span>${fmt(inputs.net_income)}</span></div>
    <div class="fund-row"><span>Revenue</span><span>${fmt(inputs.revenue)}</span></div>
    <div class="fund-row"><span>Equity</span><span>${fmt(inputs.equity)}</span></div>
    <div class="fund-row"><span>Liabilities</span><span>${fmt(inputs.liabilities)}</span></div>
  `;

  document.getElementById("fundamental-body").innerHTML = rows + inputsHtml;
  document.getElementById("fundamental-note").textContent = data.note || "";
};

const loadFundamentals = async (ticker) => {
  if (!ticker) return;
  document.getElementById("fundamental-selected").textContent = ticker;
  document.getElementById("fundamental-body").innerHTML = '<div class="empty">Loading...</div>';
  try {
    const data = await fetch(`/api/fundamentals/${ticker}`).then((r) => r.json());
    renderFundamentals(data);
  } catch (err) {
    document.getElementById("fundamental-body").innerHTML =
      '<div class="empty">Gagal mengambil data fundamental.</div>';
  }
};

const bindTableClicks = () => {
  ["scalping-table", "swing-table"].forEach((id) => {
    const container = document.getElementById(id);
    if (!container) return;
    container.addEventListener("click", (event) => {
      const row = event.target.closest("tr[data-ticker]");
      if (!row) return;
      const ticker = row.getAttribute("data-ticker");
      loadFundamentals(ticker);
    });
  });
};

async function refresh() {
  const [scalping, swing, corporate] = await Promise.all([
    fetch("/api/scalping").then((r) => r.json()),
    fetch("/api/swing").then((r) => r.json()),
    fetch("/api/corporate-actions").then((r) => r.json()),
  ]);

  document.getElementById("scalping-updated").textContent = scalping.updated_at || "-";
  document.getElementById("swing-updated").textContent = swing.updated_at || "-";
  document.getElementById("scalping-error").textContent = scalping.error || "";
  document.getElementById("swing-error").textContent = swing.error || "";
  document.getElementById("scalping-table").innerHTML = buildTable(scalping.items, "scalping");
  document.getElementById("swing-table").innerHTML = buildTable(swing.items, "swing");

  document.getElementById("ca-updated").textContent = corporate.updated_at || "-";
  document.getElementById("ca-error").textContent = corporate.error || "";
  document.getElementById("ca-table").innerHTML = buildCorporateTable(corporate.items);
}

refresh();
bindTableClicks();
setInterval(refresh, window.UI_POLL_SECONDS ? window.UI_POLL_SECONDS * 1000 : 1000);
