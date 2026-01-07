import pandas as pd
import yfinance as yf
import requests
import io
import time
from tabulate import tabulate

def fetch_real_top_10():
    url = "https://raw.githubusercontent.com/carmensyva/list-emiten/refs/heads/main/all.csv"
    
    try:
        response = requests.get(url)
        df_emiten = pd.read_csv(io.StringIO(response.text))
        tickers = df_emiten.iloc[:, 0].tolist()

        all_candidates = []
        
        print(f"Memulai pemindaian menyeluruh terhadap {len(tickers)} emiten...")
        print("Sedang menganalisa data (ini butuh waktu beberapa menit)...")

        for symbol in tickers:
            ticker_symbol = str(symbol).strip()
            ticker_jk = f"{ticker_symbol}.JK"
            
            try:
                # Ambil data sedikit lebih banyak untuk akurasi rata-rata
                df = yf.download(ticker_jk, period="15d", interval="1d", progress=False, auto_adjust=True)
                
                if df.empty or len(df) < 10:
                    continue

                close_s = df['Close'].squeeze()
                vol_s = df['Volume'].squeeze()
                
                last_close = float(close_s.iloc[-1])
                prev_close = float(close_s.iloc[-2])
                change = ((last_close - prev_close) / prev_close) * 100
                
                last_vol = float(vol_s.iloc[-1])
                avg_vol_5d = float(vol_s.iloc[-6:-1].mean())

                # Filter awal: Hanya yang naik harga & volume-nya
                if change > 0 and last_vol > avg_vol_5d:
                    vwap_val = (close_s.tail(10) * vol_s.tail(10)).sum() / vol_s.tail(10).sum()
                    vwap_scr = (last_close / vwap_val) - 1
                    vol_spike = last_vol / avg_vol_5d # Rasio kenaikan volume

                    candidate = {
                        "Date": pd.Timestamp.now().strftime('%Y-%m-%d'),
                        "Ticker": ticker_symbol,
                        "Close": last_close,
                        "Change": change,
                        "10dCnt": len(df),
                        "Status": "Akum",
                        "Type": "FV",
                        "VWAP": vwap_val,
                        "VWAPScr": vwap_scr,
                        "VolSpike": vol_spike # Kita gunakan ini untuk sortir
                    }
                    all_candidates.append(candidate)
                
                # Jeda sangat singkat agar lebih cepat tapi tidak kena ban
                time.sleep(0.05) 

            except Exception:
                continue

        # --- PROSES SORTING (TOP 10) ---
        # Kita sortir berdasarkan kenaikan harga tertinggi (Change) 
        # ATAU berdasarkan ledakan volume (VolSpike). Di sini saya pakai Change.
        top_10 = sorted(all_candidates, key=lambda x: x['Change'], reverse=True)[:10]

        # Format ulang untuk Tabel
        table_rows = []
        for item in top_10:
            table_rows.append([
                item["Date"], item["Ticker"], f"{item['Close']:,.1f}", 
                f"{item['Change']:+.2f}%", item["10dCnt"], item["Status"], 
                item["Type"], f"{item['VWAP']:,.2f}", f"{item['VWAPScr']:+.2f}", ""
            ])

        headers = ["Date", "Ticker", "Close", "Change", "10dCnt", "Status", "Type", "VWAP", "VWAPScr", "Sizing"]
        print("\n" + "="*80)
        print(" HASIL SCANNING: TOP 10 EMITEN (PRICE & VOLUME UP)")
        print("="*80)
        print(tabulate(table_rows, headers=headers, tablefmt="psql"))

    except Exception as e:
        print(f"Error Utama: {e}")

if __name__ == "__main__":
    fetch_real_top_10()
