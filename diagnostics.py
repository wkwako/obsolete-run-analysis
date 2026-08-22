import datetime

def count_runs(game_id, data):
    """For diagnostic use only. Given a game_id, counts the number of
        runs submitted per user and sorts them into buckets."""
    count_runs = {}

    for game, user_dict in data.items():
        for user, category_dict in user_dict.items():
            total = 0
            for category, runs_list in category_dict.items():
                total += len(runs_list)
            count_runs[user] = total


    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11+": 0}
    for user, total in count_runs.items():
        if total == 1:
            buckets["1"] += 1
        elif total <= 5:
            buckets["2-5"] += 1
        elif total <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1

    print(buckets)

def describe_game(self, game_id):
    """Quick diagnostic: size and balance of a game's examples table."""
    df = self.repo.load_from_db(game_id)

    if df.empty:
        print(f"No data for game {game_id}")
        return

    print(f"--- {game_id} ---")
    print(f"total examples:     {len(df)}")
    print(f"distinct runners:   {df['user_id'].nunique()}")
    print(f"class balance:      {df['label'].mean():.3f} retained")

    # examples-per-cutoff, to see where the data concentrates
    per_cutoff = df.groupby("cutoff").size()
    print(f"cutoffs:            {len(per_cutoff)}")
    print(f"examples per cutoff:\n{per_cutoff}")

def retention_diagnostic(data, game_id, cutoff_str, min_prior_runs=3, window_months=12):
    """
    Measures class balance for the retention label among qualifying runners.

    game_id:        the game to analyze (must already be cached)
    cutoff_str:     cutoff date T as "YYYY-MM-DD"
    min_prior_runs: minimum runs BEFORE T for a runner to qualify (need history for features)
    window_months:  how many months after T defines the retention window
    """
    # --- load cached data ---
    #data = self.load_run_data(game_id)

    # --- parse the cutoff and window end ---
    cutoff = datetime.datetime.fromisoformat(cutoff_str)
    window_end = cutoff + datetime.timedelta(days=window_months * 30)  # approx months

    # --- counters ---
    total_runners = 0
    qualifying = 0          # runners with >= min_prior_runs before cutoff
    retained = 0            # qualifying runners active in [cutoff, window_end]
    churned = 0             # qualifying runners NOT active in that window
    skipped_no_dates = 0    # runners we couldn't place in time at all

    # data is {game: {user: {category: [runs]}}}
    for game_key, user_dict in data.items():
        for user, category_dict in user_dict.items():
            total_runners += 1

            # flatten this user's run dates across ALL categories
            run_dates = []
            for category, runs_list in category_dict.items():
                for run in runs_list:
                    d = run.get("date")
                    if d:  # skip null dates
                        try:
                            run_dates.append(datetime.datetime.fromisoformat(d))
                        except ValueError:
                            pass  # skip malformed dates

            if not run_dates:
                skipped_no_dates += 1
                continue

            # count runs strictly before the cutoff
            prior_runs = [d for d in run_dates if d < cutoff]
            if len(prior_runs) < min_prior_runs:
                continue  # not enough history to qualify

            qualifying += 1

            # label: did they submit anything in [cutoff, window_end]?
            active_in_window = any(cutoff <= d <= window_end for d in run_dates)
            if active_in_window:
                retained += 1
            else:
                churned += 1

    # --- report ---
    print(f"\n{'='*55}")
    print(f"RETENTION DIAGNOSTIC — game {game_id}")
    print(f"Cutoff T = {cutoff_str}, window = {window_months} months, min prior runs = {min_prior_runs}")
    print(f"{'='*55}")
    print(f"Total runners in game:        {total_runners}")
    print(f"Skipped (no usable dates):    {skipped_no_dates}")
    print(f"Qualifying runners (>= {min_prior_runs} prior): {qualifying}")
    if qualifying == 0:
        print("No qualifying runners — try an earlier cutoff or lower min_prior_runs.")
        return
    ret_pct = 100 * retained / qualifying
    chu_pct = 100 * churned / qualifying
    print(f"  Retained: {retained} ({ret_pct:.1f}%)")
    print(f"  Churned:  {churned} ({chu_pct:.1f}%)")
    print(f"{'='*55}\n")

    return {"qualifying": qualifying, "retained": retained, "churned": churned}