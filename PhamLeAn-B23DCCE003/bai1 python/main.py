import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Khởi tạo WebDriver ---
driver = webdriver.Chrome()
players = {}  # dict to store player data; keys are (name, team)

# --- Bảng 1: Tổng quan ---
print("🔄 Bắt đầu lấy dữ liệu bảng Tổng quan...")
driver.get('https://fbref.com/en/comps/9/stats/Premier-League-Stats')
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'stats_standard')))
tbody = driver.find_element(By.ID, 'stats_standard').find_element(By.TAG_NAME, 'tbody')
for row in tbody.find_elements(By.TAG_NAME, 'tr'):
    if 'thead' in row.get_attribute('class'):
        continue
    try:
        minutes = row.find_element(By.CSS_SELECTOR, "td[data-stat='minutes']").text.strip()
        if not minutes or int(minutes.replace(',', '')) < 90:
            continue
    except:
        continue
    name = row.find_element(By.CSS_SELECTOR, "td[data-stat='player']").text.strip() or 'N/a'
    team = row.find_element(By.CSS_SELECTOR, "td[data-stat='team']").text.strip() or 'N/a'
    key = (name, team)
    # Nation extraction
    nat_cell = row.find_element(By.CSS_SELECTOR, "td[data-stat='nationality']")
    try:
        span = nat_cell.find_element(By.TAG_NAME, 'span')
        nation = span.text.strip().split()[-1]
    except:
        nation = 'N/a'
    # Initialize entry
    players[key] = {'Name': name, 'Team': team, 'Nation': nation}
    # Stats from Table 1
    for col, stat in {
        'Position': 'position', 'Age': 'birth_year', 'Matches_played': 'games', 'Starts': 'games_starts',
        'Minutes': 'minutes', 'Goals': 'goals', 'Assists': 'assists', 'Yellow Cards': 'cards_yellow', 'Red Cards': 'cards_red',
        'Expected_xG': 'xg', 'Expected_xAG': 'xg_assist', 'PrgC': 'progressive_carries', 'PrgP': 'progressive_passes',
        'PrgR': 'progressive_passes_received', 'Per90_Gls': 'goals_per90', 'Per90_Ast': 'assists_per90',
        'Per90_xG90': 'xg_per90', 'Per90_xAG': 'xg_assist_per90'
    }.items():
        try:
            val = row.find_element(By.CSS_SELECTOR, f"td[data-stat='{stat}']").text.strip() or 'N/a'
        except:
            val = 'N/a'
        if col == 'Age' and val.isdigit():
            players[key][col] = str(2025 - int(val))
        else:
            players[key][col] = val

# Helper to process subsequent tables
def process_table(url, table_id, stats_dict, table_name):
    print(f"🔄 Bắt đầu lấy dữ liệu bảng {table_name}...")
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, table_id)))
    tbody = driver.find_element(By.ID, table_id).find_element(By.TAG_NAME, 'tbody')
    for row in tbody.find_elements(By.TAG_NAME, 'tr'):
        if 'thead' in row.get_attribute('class'):
            continue
        name = row.find_element(By.CSS_SELECTOR, "td[data-stat='player']").text.strip() or 'N/a'
        team = row.find_element(By.CSS_SELECTOR, "td[data-stat='team']").text.strip() or 'N/a'
        key = (name, team)
        if key not in players:
            continue
        for col, stat in stats_dict.items():
            try:
                players[key][col] = row.find_element(By.CSS_SELECTOR, f"td[data-stat='{stat}']").text.strip() or 'N/a'
            except:
                players[key][col] = 'N/a'

# Subsequent tables
process_table('https://fbref.com/en/comps/9/keepers/Premier-League-Stats', 'stats_keeper',
              {'GA90': 'gk_goals_against_per90', 'Save%': 'gk_save_pct', 'Penalty_Save%': 'gk_pens_save_pct', 'CS%': 'gk_clean_sheets_pct'},
              'Thủ môn')
process_table('https://fbref.com/en/comps/9/shooting/Premier-League-Stats', 'stats_shooting',
              {'SoT%': 'shots_on_target_pct', 'SoT_per90': 'shots_on_target_per90', 'G_sh': 'goals_per_shot', 'Dist': 'average_shot_distance'},
              'Dứt điểm')
process_table('https://fbref.com/en/comps/9/passing/Premier-League-Stats', 'stats_passing',
              {'Cmp': 'passes_completed', 'Total_Cmp%': 'passes_pct', 'TotDist': 'passes_total_distance',
               'Cmp_Short%': 'passes_pct_short', 'Cmp_Median%': 'passes_pct_medium', 'Cmp_Long%': 'passes_pct_long',
               'KP': 'assisted_shots', '1/3_Expected': 'passes_into_final_third', 'PPA': 'passes_into_penalty_area',
               'CrsPA': 'crosses_into_penalty_area', 'PrgP_Expected': 'progressive_passes'},
              'Chuyền bóng')
process_table('https://fbref.com/en/comps/9/gca/Premier-League-Stats', 'stats_gca',
              {'SCA': 'sca', 'SCA90': 'sca_per90', 'GCA': 'gca', 'GCA90': 'gca_per90'},
              'GCA')
process_table('https://fbref.com/en/comps/9/defense/Premier-League-Stats', 'stats_defense',
              {'Tkl': 'tackles', 'TklW': 'tackles_won', 'Att_C': 'challenges', 'Lost_C': 'challenges_lost',
               'Blocks': 'blocks', 'Sh': 'blocked_shots', 'Pass': 'blocked_passes', 'Int': 'interceptions'},
              'Phòng ngự')
process_table('https://fbref.com/en/comps/9/possession/Premier-League-Stats', 'stats_possession',
              {'Touches': 'touches', 'DefPen_': 'touches_def_pen_area', 'Def3rd': 'touches_def_3rd', 'Mid_3rd': 'touches_mid_3rd',
               'Att_3rd': 'touches_att_3rd', 'Att_pen': 'touches_att_pen_area', 'Take-Ons_Att': 'take_ons',
               'Succ%': 'take_ons_won_pct', 'Tkld%': 'take_ons_tackled_pct', 'Carries': 'carries',
               'ProDist': 'carries_progressive_distance', 'PrgC_cr': 'progressive_carries', '1/3_Cr': 'carries_into_final_third',
               'CPA': 'carries_into_penalty_area', 'Mis': 'miscontrols', 'Dis': 'dispossessed', 'Rec': 'passes_received',
               'PrgR_R': 'progressive_passes_received'},
              'Kiểm soát bóng')
process_table('https://fbref.com/en/comps/9/misc/Premier-League-Stats', 'stats_misc',
              {'Fls': 'fouls', 'Fld': 'fouled', 'Off': 'offsides', 'Crs': 'crosses',
               'Recov': 'ball_recoveries', 'Won': 'aerials_won', 'Lost': 'aerials_lost', 'Won%': 'aerials_won_pct'},
              'Khác')

# --- Hoàn thành crawl ---
print("✅ Hoàn thành crawl. Đang xuất file CSV...")
driver.quit()

# Xuất CSV
fieldnames = [
    'Name', 'Team', 'Nation',
    'Position', 'Age', 'Matches_played', 'Starts', 'Minutes', 'Goals', 'Assists', 'Yellow Cards', 'Red Cards',
    'Expected_xG', 'Expected_xAG', 'PrgC', 'PrgP', 'PrgR', 'Per90_Gls', 'Per90_Ast', 'Per90_xG90', 'Per90_xAG',
    'GA90', 'Save%', 'CS%', 'Penalty_Save%', 'SoT%', 'SoT_per90', 'G_sh', 'Dist',
    'Cmp', 'Total_Cmp%', 'TotDist', 'Cmp_Short%', 'Cmp_Median%', 'Cmp_Long%', 'KP', '1/3_Expected', 'PPA', 'CrsPA', 'PrgP_Expected',
    'SCA', 'SCA90', 'GCA', 'GCA90',
    'Tkl', 'TklW', 'Att_C', 'Lost_C', 'Blocks', 'Sh', 'Pass', 'Int',
    'Touches', 'DefPen_', 'Def3rd', 'Mid_3rd', 'Att_3rd', 'Att_pen', 'Take-Ons_Att', 'Succ%', 'Tkld%', 'Carries', 'ProDist', 'PrgC_cr', '1/3_Cr', 'CPA', 'Mis', 'Dis', 'Rec', 'PrgR_R',
    'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost', 'Won%'
]
print(f"Total columns: {len(fieldnames)}")
with open('results.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, restval='N/a')
    writer.writeheader()
    for data in sorted(players.values(), key=lambda x: (x['Name'], x['Team'])):
        writer.writerow(data)
print('✅ File saved: results.csv')