# 🎮 HUNT - Easy Setup Guide for Windows

**A visual simulation where you watch predators and prey evolve hunting and survival strategies in real-time.**

No coding required! Just follow these steps to watch evolution in action on your Windows computer.

---

## 📋 What You'll Need

- **Windows 10 or 11**
- **About 500 MB of free disk space**
- **15 minutes of setup time**

That's it! This works on regular laptops and desktops - you don't need a gaming PC.

---

## 🚀 Step-by-Step Setup

### Step 1: Install Python

Python is the programming language that runs HUNT. Don't worry - you won't need to write any code!

1. **Go to**: https://www.python.org/downloads/
2. **Click** the big yellow button that says "Download Python 3.XX.X"
3. **Run the installer** (the file you just downloaded)
4. **⚠️ IMPORTANT**: Check the box that says **"Add Python to PATH"** at the bottom of the installer
5. **Click** "Install Now"
6. **Wait** for it to finish (about 2-3 minutes)
7. **Click** "Close" when done

**How to check if it worked:**
- Press `Windows Key + R`
- Type `cmd` and press Enter
- In the black window, type: `python --version`
- You should see something like "Python 3.12.1"
- Type `exit` to close the window

---

### Step 2: Download HUNT

1. **Go to**: https://github.com/DarrellThomas/hunt
2. **Click** the green `<> Code` button near the top
3. **Click** "Download ZIP"
4. **Save the file** somewhere you can find it (like your Desktop or Downloads folder)
5. **Right-click** the ZIP file and select "Extract All..."
6. **Choose** where to extract it (your Desktop is fine)
7. **Click** "Extract"

You should now have a folder called `hunt-master` on your Desktop.

---

### Step 3: Install Required Programs

HUNT needs a few helper programs to work. We'll install them automatically!

1. **Open the folder** you just extracted (`hunt-master`)
2. **Right-click** on an empty space in the folder while holding **Shift**
3. **Select** "Open PowerShell window here" or "Open Command window here"
   - If you see a blue window (PowerShell) or black window (Command Prompt), you did it right!

4. **Copy and paste** this command (press Enter after pasting):

```
pip install numpy pygame matplotlib
```

5. **Wait** while it downloads and installs (1-2 minutes)
   - You'll see lots of text scrolling - this is normal!
   - When you see a blinking cursor again, it's done

**If you see an error** saying "pip is not recognized":
- Close the window
- Restart your computer
- Try Step 3 again (Python PATH needs a restart to work)

---

### Step 4: Run HUNT! 🎉

Still in that PowerShell/Command window:

1. **Type** this command and press Enter:

```
python run.py
```

2. **Wait** about 5-10 seconds while it starts
3. **A window should appear** showing green dots (prey) and red dots (predators)!

---

## 🎮 How to Use It

Once the window opens:

- **Watch**: Green prey try to escape, red predators try to hunt
- **Press SPACEBAR**: Pause or resume the simulation
- **Press S**: Save statistics (creates a data file)
- **Press ESC**: Close the program

### What You're Seeing

- **Green dots** = Prey (trying to survive)
- **Red dots** = Predators (trying to hunt)
- **Bottom panel** = Shows population counts and statistics

The longer you watch, the better they get at surviving and hunting! This is evolution happening in real-time.

---

## ⚙️ Adjusting for Slower Computers

If the simulation is slow or choppy on your computer:

1. **Open** the `src` folder inside `hunt-master`
2. **Right-click** on `config.py` and open with Notepad
3. **Find** the line that says `PREY_MAX_SPEED = 3.0` (near the top)
4. **Scroll down** to find these lines and change the numbers:

```python
# Make these numbers smaller:
PREY_MAX_SPEED = 2.0              # Changed from 3.0
PRED_MAX_SPEED = 1.5              # Changed from 2.5
```

5. **Save** the file (Ctrl+S)
6. **Run** HUNT again

**Still slow?** In that same file, scroll way down to find:
```python
initial_prey=200
initial_predators=40
```

Change them to:
```python
initial_prey=50
initial_predators=10
```

This runs fewer agents (dots) so it's faster on slower computers.

---

## 🐛 Troubleshooting

### "python is not recognized"
**Fix**: Python isn't installed correctly.
- Uninstall Python (Windows Settings → Apps)
- Reinstall it, making sure to check "Add Python to PATH"
- Restart your computer

### "No module named pygame"
**Fix**: The required programs didn't install.
- Open PowerShell/Command Prompt in the hunt-master folder again
- Run: `pip install numpy pygame matplotlib` again
- Make sure you're connected to the internet

### The window is too big for my screen
**Fix**: Make the window smaller.
1. Open `src/config.py` in Notepad (see "Adjusting for Slower Computers" above)
2. Find the very bottom of the file where it says:
```python
world = World(width=1600, height=1200, ...)
```
3. Change it to:
```python
world = World(width=800, height=600, ...)
```
4. Save and run again

### The simulation is too fast / too slow
**Fix**: Adjust the speed.
- **Press SPACEBAR** to pause
- Close HUNT (press ESC)
- Open `src/main.py` in Notepad
- Find the line near the bottom: `fps=30`
- Change it to `fps=15` (slower) or `fps=60` (faster)
- Save and run again

### Everything is working but I want more/fewer creatures
**Fix**: Adjust population sizes.
- Open `src/config.py` in Notepad
- Scroll all the way to the bottom
- Find: `initial_prey=200, initial_predators=40`
- Change to smaller numbers for fewer creatures, larger for more
- **Recommended ranges**:
  - Prey: 20-500
  - Predators: 5-100
- Save and run again

---

## 📊 Viewing Your Data

After running HUNT and pressing 'S' to save:

1. **Open** PowerShell/Command Prompt in the hunt-master folder
2. **Type**: `cd src` and press Enter
3. **Type**: `python analyze_evolution.py` and press Enter
4. **Wait** about 10 seconds
5. **A chart will appear** showing how populations changed over time!
6. **Look** in the `results` folder for saved images

---

## ❓ Frequently Asked Questions

**Q: How long should I let it run?**
A: At least 5-10 minutes to see evolution happening. The longer you run it, the more interesting it gets!

**Q: Can I change the colors?**
A: Yes! Advanced users can edit `src/main.py` and change the color values (look for `prey_color` and `predator_color`).

**Q: Will this harm my computer?**
A: No! It's perfectly safe. It's like playing a simple video game - your computer can handle it.

**Q: Can I run multiple simulations at once?**
A: Yes! Just open a new PowerShell window and run `python run.py` again. Each window is independent.

**Q: Where are the saved files?**
A: Look in the `results` folder inside `hunt-master`. You'll find statistics (.npz files) and charts (.png images).

**Q: Can I share my results?**
A: Absolutely! The .png images in the `results` folder are regular pictures you can share anywhere.

---

## 🎓 What's Actually Happening?

Each dot (agent) has a tiny "brain" (neural network) that controls how it moves. When prey survive long enough, they create offspring with slightly different brains. When predators successfully hunt, they also reproduce. Over time, the successful strategies survive and the poor strategies die out.

This is **neuroevolution** - evolution of neural networks through natural selection. You're watching artificial intelligence evolve in real-time!

---

## 🔗 Want to Learn More?

- **Read** `docs/THESIS.md` to understand the science
- **Read** `docs/QUICKSTART.md` for more advanced options
- **Read** `docs/RESULTS.md` to see what the researchers discovered

---

## 🆘 Still Need Help?

If these instructions didn't work:

1. **Check** that you're using Windows 10 or 11
2. **Make sure** you extracted the ZIP file (don't run from inside the ZIP)
3. **Try** restarting your computer and starting from Step 3
4. **Ask** a tech-savvy friend to help with the Python installation

---

## 🎉 You Did It!

Congratulations! You're now running a real scientific simulation on your computer. Watch those creatures evolve, experiment with different settings, and share your results!

**Enjoy watching evolution in action! 🧬**

---

*Made by researchers who believe science should be accessible to everyone.*
