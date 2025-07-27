# Scrctrl-WearOS

### Like Scrcpy, but for WearOS devices

> **Note:** This project is currently in **BETA**.

------

## Quick Start

### Prerequisites

- You **must have ADB installed** on your system.
   Follow the installation guide here:
   [Installing ADB on Windows, macOS, and Linux](https://www.xda-developers.com/install-adb-windows-macos-linux/)

------

### Setup Virtual Environment

Create a virtual environment:

```
python3 -m venv venv
```

Activate the virtual environment:

- **Windows:**

  ```
  venv\Scripts\activate.bat
  ```

- **Linux/macOS:**

  ```
  source ./venv/bin/activate
  ```

------

### Install Dependencies

```
pip install -r requirements.txt
```

------

## Running the Application

Start the app by running:

Enable Developer settings then enable wireless debugging then use code.

```
adb pair <ip>:<port>
adb connect <ip>:<port>

python3 scrctrl.py
```

The GUI will launch automatically.