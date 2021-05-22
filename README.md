# FEM2D Implementation

## Requirements

- Python3
- Pip 20+

## Usage
### Virtual Environment
#### On Unix Systems:
```bash
python3 -m venv venv
```

#### On Windows:
```powershell
python -m venv venv
```

### Active the Virtual Environment

#### On Unix Systems:
```bash
source venv/bin/activate
```

#### On Windows:
```powershell
.\venv\Scripts\activate
```

This should leave you in a bash console with the prompt `(venv): [<user>]$ `

### Install dependencies
Inside the venv session, run the following line
```bash
pip install -r requirements.txt
```

### Run the code
Inside the venv session, run the following line
```bash
python -m application <data_file>
```
where <data_file> is the .dat file with the input information. 

*Note: do not add the `.dat` extension*
