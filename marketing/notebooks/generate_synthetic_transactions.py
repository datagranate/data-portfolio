import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path.cwd()  
csv_path = base_dir.parent / "data" / "processed" / "marketing_clean.csv"
marketing_clean = pd.read_csv(csv_path)


# generate a synthetic transaction dataset from the (aggregated) clean marketing dataset
# assumes fields as in marketing_clean.csv (see marketing_cleaning.ipynb)
# reference_date, the notional 'today' was derived in marketing_cleaning.ipynb
# omitting various safety checks temporarily as the dataset is known
# aggregated data for only the previous two years per data dictionary

current_date = pd.Timestamp('2014-10-05')
clip_start = current_date - pd.Timedelta(days=365*2)

def generate_transaction_dataset(
    df, 
    reference_date=current_date,
    clip_start=clip_start,
    early_burst_ratio_base = 0.2, # 20% of transactions happen in early burst window
    early_window_days_base = 60, # food retail, so fixed days
    spend_skew_sigma = 0.25 # log-normal, low-medium skew  
    ):
    
    np.random.seed(42) 
    transactions = []    

    for _, row in df.iterrows():
        cust_id = row['ID']
        n_purchases = row['NumPurchases']
        total_spend = row['TotalSpend']
        tenure_days = row['DaysSinceJoin']
        last_purchase_days_ago = row['Recency']

        if n_purchases == 0 or total_spend == 0:
            continue
                
        # time boundaries 
        join_date = reference_date - pd.Timedelta(days=tenure_days)
        start_date = max(join_date, clip_start)
        actual_last_purchase_date = reference_date - pd.Timedelta(days=last_purchase_days_ago)
        
        # duration T, full potential window of activity
        T = (actual_last_purchase_date - start_date).days
        if T <= 0: # should never happen with  this clean dataset
            continue

        # adjust early window for clipped (customers joining before clip_start)
        if join_date >= clip_start:
            early_days = early_window_days_base
            early_ratio = early_burst_ratio_base
            is_clipped = False
        else:
            # joined before clip_start: reduce early days by the overhang 
            overhang = (clip_start - join_date).days
            early_days = max(0, early_window_days_base - overhang)
            # reduce early purchase ratio proportionally
            early_ratio = early_burst_ratio_base * (early_days / early_window_days_base) if early_window_days_base > 0 else 0.0
            is_clipped = True

        # ensure early days are not more than full activity window
        E = min(early_days, T)

        # purchase split
        if E <= 0:
            n_early = 0
            n_late = n_purchases
        else:
            n_early = int(np.round(n_purchases * early_ratio))
            n_early = max(0, min(n_early, n_purchases))
            n_late = n_purchases - n_early

        # generate offsets (days_ago relative to start_date) ---
        early_dates = np.array([], dtype=float)
        late_dates = np.array([], dtype=float)

        if n_early > 0:
            # offsets in [0, E)
            if E <= 1:
                early_dates = np.zeros(n_early, dtype=float)
            else:
                early_dates = np.random.uniform(0, E, n_early)

        if n_late > 0:
            # offsets in [E, T)
            if E >= T - 1:
                late_dates = np.array([T - 1.0] * n_late, dtype=float)
            else:
                late_dates = np.random.uniform(E, T, n_late)

        all_days_ago = np.concatenate([early_dates, late_dates])
        if all_days_ago.size == 0:
            continue

        all_days_ago = np.sort(all_days_ago)

        # anchor first purchase to start_date ONLY if start_date is NOT clipped 
        if (not is_clipped) and n_purchases >= 2:
            all_days_ago[0] = 0.0
            all_days_ago = np.sort(all_days_ago)

        # force last date to match actual_last_purchase_date
        target_offset = (actual_last_purchase_date - start_date).days
        all_days_ago[-1] = float(target_offset)

        transaction_dates = [start_date + pd.Timedelta(days=float(d)) for d in all_days_ago]

        # generate purchase amounts
        if n_purchases == 1:
            spend_final = np.array([total_spend], dtype=float)
        else:
            mean_spend = total_spend / n_purchases
            spend_raw = np.random.lognormal(mean=np.log(mean_spend), sigma=spend_skew_sigma, size=n_purchases)
            spend_final = spend_raw * (total_spend / spend_raw.sum())
            np.random.shuffle(spend_final)

        #build transactions
        for i in range(n_purchases):
            transactions.append({
                'ID': cust_id,
                'transaction_date': transaction_dates[i].date(),
                'amount': round(float(spend_final[i]), 2),
                'purchase_index': i + 1
            })

    return pd.DataFrame(transactions)

synthetic_transactions = generate_transaction_dataset(marketing_clean)

out_path = base_dir.parent / "data" / "processed" / "synthetic_transactions.csv"
synthetic_transactions.to_csv(out_path, index=False)