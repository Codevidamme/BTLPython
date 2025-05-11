from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import time
import pandas as pd


def load_filtered_players(csv_path):
    """Load and filter players with more than 900 Minutes of play time."""
    try:
        stats_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại đường dẫn: {csv_path}")
        return set() # Trả về set rỗng nếu không tìm thấy file

    # Kiểm tra xem cột 'Minutes' và 'Name', 'Team' có tồn tại không
    if 'Minutes' not in stats_df.columns:
        print("LỖI: File CSV không có cột 'Minutes'.")
        return set()
    if 'Name' not in stats_df.columns:
        print("LỖI: File CSV không có cột 'Name'.")
        return set()
    if 'Team' not in stats_df.columns:
        print("LỖI: File CSV không có cột 'Team'.")
        return set()

    # --- PHẦN SỬA LỖI CHÍNH ---
    # 1. Đảm bảo cột 'Minutes' được xử lý như chuỗi để có thể dùng .str.replace()
    stats_df['Minutes'] = stats_df['Minutes'].astype(str)
    
    # 2. Loại bỏ dấu phẩy (,) trong cột 'Minutes'
    stats_df['Minutes'] = stats_df['Minutes'].str.replace(',', '', regex=False)
    
    # 3. Chuyển cột 'Minutes' sang dạng số, các giá trị không hợp lệ sẽ thành NaN
    stats_df['Minutes'] = pd.to_numeric(stats_df['Minutes'], errors='coerce')
    
    # 4. (Tùy chọn nhưng nên có) Loại bỏ các hàng có giá trị NaN ở cột 'Minutes' sau khi chuyển đổi
    # Điều này đảm bảo chỉ các giá trị số hợp lệ được xem xét.
    stats_df.dropna(subset=['Minutes'], inplace=True)
    # --- KẾT THÚC PHẦN SỬA LỖI ---

    # Bây giờ, lọc dựa trên cột 'Minutes' đã là số
    filtered_df = stats_df[stats_df['Minutes'] > 900]
    
    # Lấy cột 'Name', 'Team' và loại bỏ các hàng trùng lặp
    unique_filtered_players = filtered_df[['Name', 'Team']].drop_duplicates()
    
    # Tạo set các tuple (name, team)
    # .str.strip() để loại bỏ khoảng trắng thừa ở đầu/cuối tên và đội bóng (nếu có)
    filtered_players_set = set(zip(unique_filtered_players['Name'].str.strip(), 
                                   unique_filtered_players['Team'].str.strip()))
    
    if not filtered_players_set:
        print("Không tìm thấy cầu thủ nào có > 900 phút thi đấu sau khi lọc.")
    else:
        print(f"Đã tìm thấy {len(filtered_players_set)} cầu thủ/đội duy nhất có > 900 phút thi đấu.")
        
    return filtered_players_set

# --- CÁCH SỬ DỤNG ---
# Giả sử file results.csv của bạn nằm cùng thư mục với file Python này
# hoặc bạn cung cấp đường dẫn đầy đủ.
# csv_file_path = 'results.csv' 
# players_set = load_filtered_players(csv_file_path)

# In ra một vài cầu thủ để kiểm tra (nếu có)
# if players_set:
#     print("\nMột vài cầu thủ (và đội) đã lọc được:")
#     count = 0
#     for player, team in players_set:
#         print(f"- {player} ({team})")
#         count += 1
#         if count >= 5: # In ra tối đa 5 cầu thủ
#             break


def setup_driver():
    """Initialize and setup the Chrome driver."""
    driver = uc.Chrome()
    wait = WebDriverWait(driver, 10)
    return driver, wait


def close_cookie_popups(driver):
    """Attempt to close any cookie consent popups."""
    try:
        cookie_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'Cookie') or contains(text(), 'Close')]")
        for button in cookie_buttons:
            driver.execute_script("arguments[0].click();", button)
            time.sleep(0.5)
    except:
        pass


def navigate_to_page(driver, url):
    """Navigate to a specific URL and wait for page to load."""
    driver.get(url)
    time.sleep(3)
    driver.execute_script("window.scrollBy(0, 300);")
    time.sleep(1)


def get_team_elements(driver, wait):
    """Get all team elements from the league table."""
    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'tbody')))
    teams_table = driver.find_element(By.TAG_NAME, 'tbody')
    return teams_table.find_elements(By.TAG_NAME, 'tr')


def get_team_url(team):
    """Extract team URL from team element."""
    try:
        team_link = team.find_element(By.TAG_NAME, 'a')
        return team_link.get_attribute('href')
    except Exception as e:
        print(f"Could not find team link: {str(e)}")
        return None


def extract_player_data(player, team_name, filtered_players_set):
    """Extract player data from a player element."""
    try:
        player_name_elem = player.find_element(By.XPATH, ".//th[@class='td-player']//a")
        player_name = player_name_elem.text.strip()
        
        if (player_name, team_name) not in filtered_players_set:
            print(f"Skip {player_name}")
            return None
        
        try:
            transfer_value = player.find_element(By.CLASS_NAME, 'player-tag').text
        except:
            try:
                transfer_value = player.find_element(By.XPATH, ".//td[contains(@class, 'value')]").text
            except:
                transfer_value = "Not available"
        
        print(f"  - Added player: {player_name}, Value: {transfer_value}")
        return {
            'Team': team_name,
            'Player': player_name,
            'Transfer Value': transfer_value
        }
    except Exception as e:
        print(f"E: {str(e)}")
        return None


def get_players_data(driver, wait, team_name, filtered_players_set):
    """Get data for all players of a team."""
    player_data = []
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'tbody')))
        
        players_table = driver.find_element(By.TAG_NAME, 'tbody')
        players = players_table.find_elements(By.TAG_NAME, 'tr')
        
        print(f"  Found {len(players)} players")
        
        for player in players:
            player_info = extract_player_data(player, team_name, filtered_players_set)
            if player_info:
                player_data.append(player_info)
    except Exception as e:
        print(f"Error finding players: {str(e)}")
    
    return player_data


def process_team(driver, wait, team, filtered_players_set, main_url):
    """Process a single team to extract player data."""
    team_name = team.find_elements(By.TAG_NAME, 'td')[2].text
    print(f"Processing team: {team_name}")
    
    team_url = get_team_url(team)
    if team_url:
        navigate_to_page(driver, team_url)
    else:
        # Try JavaScript click as a fallback
        try:
            driver.execute_script("arguments[0].click();", team)
            time.sleep(3)
        except Exception as e:
            print(f"Failed to navigate to team page: {str(e)}")
            return []
    
    player_data = get_players_data(driver, wait, team_name, filtered_players_set)
    
    # Return to main page
    navigate_to_page(driver, main_url)
    
    return player_data


def scrape_player_values(csv_file, output_file):
    """Main function to scrape player transfer values."""
    # Setup
    filtered_players_set = load_filtered_players(csv_file)
    driver, wait = setup_driver()
    
    # Main page URL
    main_url = 'https://www.footballtransfers.com/us/leagues-cups/national/uk/premier-league'
    
    try:
        # Initial navigation
        navigate_to_page(driver, main_url)
        close_cookie_popups(driver)
        
        # Process all teams
        all_team_data = []
        all_teams = get_team_elements(driver, wait)
        
        for i in range(len(all_teams)):
            # Re-get elements as the DOM may have been refreshed
            all_teams = get_team_elements(driver, wait)
            team = all_teams[i]
            
            team_data = process_team(driver, wait, team, filtered_players_set, main_url)
            all_team_data.extend(team_data)
        
        # Save data
        df = pd.DataFrame(all_team_data)
        df.to_csv(output_file, index=False)
        print(f"Data saved to '{output_file}'. Total players: {len(all_team_data)}")
    
    finally:
        # Clean up
        driver.quit()


if __name__ == "__main__":
    input_csv = 'results.csv'
    output_csv = 'player_values.csv'
    scrape_player_values(input_csv, output_csv)