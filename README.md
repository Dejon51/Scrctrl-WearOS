# Scrctrl-WearOS

### Like Scrcpy, but for WearOS devices

> **Note:** This project is currently in **BETA**.

------

## Quick Start

### Prerequisites

- You **must have ADB installed** on your system.
   Follow the installation guide here:
   [Installing ADB on Windows, macOS, and Linux](https://www.xda-developers.com/install-adb-windows-macos-linux/)

- You also **Must have ffmpeg** on your system.
    Follow the installation guide here:
    [Installing ffmpeg on Windows, macOS, and Linux](https://ffmpeg.org/download.html)


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
<br>
<br>
<br>
<br>
------
```
MIT License

Copyright (c) 2025 scrpio141

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```