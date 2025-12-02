import json
import os
import sys
from bioblend.galaxy import GalaxyInstance

CONFIG_FILE = os.path.expanduser("~/zyzk/secrets/galaxy_config.json")

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 配置文件缺失: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    print(f"正在连接: {config.get('galaxy_url')} ...")
    try:
        gi = GalaxyInstance(url=config['galaxy_url'], key=config['api_key'])
        user = gi.users.get_current_user()
        print(f"✅ 认证成功 | 用户: {user.get('email')} | ID: {user.get('id')}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")

if __name__ == "__main__":
    main()
