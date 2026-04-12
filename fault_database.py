"""
PredictIQ - Fault Diagnosis Database
Parsed from industrial maintenance CSV data
"""

import csv
import io

CSV_DATA = """category,machine_type,fault,causes,diagnosis,repair,difficulty,time_min
CNC,VMC,Machine not powering on,No input power;Blown fuse;Transformer fault,Check voltage;Inspect fuse;Measure transformer,Restore power;Replace fuse;Replace transformer,Easy,30
CNC,VMC,Axis not moving,Servo drive fault;Cable issue;Mechanical jam,Check alarm;Inspect cables;Manual jog,Replace drive;Fix cable;Clear jam,Medium,120
CNC,VMC,Overheating,Coolant failure;Overload,Check coolant;Check load,Fix coolant;Reduce load,Medium,90
CNC,VMC,Poor accuracy,Backlash;Loose coupling,Measure backlash;Inspect coupling,Adjust screws;Tighten coupling,Hard,180
CNC,Lathe,Spindle not rotating,Motor failure;Drive fault,Check drive;Measure voltage,Replace motor;Repair drive,Medium,60
CNC,Lathe,Vibration,Bearing wear;Imbalance,Check bearings;Measure vibration,Replace bearing;Balance spindle,Hard,240
CNC,Lathe,Tool changer failure,Sensor issue;Air pressure low,Check sensors;Check air,Replace sensor;Fix air supply,Medium,90
CNC,5-Axis,Position error,Encoder fault;Thermal drift,Check encoder;Measure temp,Replace encoder;Recalibrate,Hard,180
CNC,5-Axis,Axis drift,Feedback issue;Loose coupling,Check feedback;Inspect coupling,Fix feedback;Tighten,Hard,150
CNC,5-Axis,Control alarm,Software fault;Drive issue,Check alarm code,Reset;Repair drive,Medium,60
Energy,VFD,Overcurrent fault,Motor short;Overload,Check motor;Check load,Repair motor;Reduce load,Medium,60
Energy,VFD,No output,IGBT failure;PCB fault,Check DC bus;Test IGBT,Replace IGBT;Repair PCB,Hard,120
Energy,VFD,Overvoltage,Input spike;Wrong setting,Measure voltage,Adjust settings;Stabilize supply,Medium,60
Energy,VFD,Overheating,Fan failure;Dust,Check fan;Clean unit,Replace fan;Clean,Easy,45
Energy,VFD,Communication error,Cable issue;Protocol mismatch,Check cable;Check settings,Replace cable;Fix config,Medium,60
Energy,Inverter,No output voltage,MOSFET failure;Battery low,Check battery;Test MOSFET,Replace MOSFET;Charge battery,Medium,60
Energy,Inverter,Not charging,Panel fault;Controller issue,Check panel voltage,Replace controller,Medium,90
Energy,Inverter,Beeping alarm,Low battery;Overload,Check load;Check battery,Reduce load;Charge battery,Easy,30
Energy,Compressor,Low efficiency,Air leak;Clogged filter,Check pressure;Inspect filter,Fix leak;Replace filter,Easy,45
Energy,Compressor,Not starting,Motor issue;Power fault,Check motor;Check supply,Repair motor;Fix wiring,Medium,60
Aerospace,5-Axis CNC,Loss of precision,Thermal drift;Calibration error,Laser measurement,Recalibrate,Expert,300
Aerospace,5-Axis CNC,Spindle overheating,Cooling failure,Check coolant,Fix cooling,Hard,180
Aerospace,Composite Machine,Improper curing,Sensor fault;Heater issue,Check temperature,Replace sensor,Hard,120
Aerospace,Composite Machine,Uneven material,Pressure issue;Alignment fault,Check pressure,Adjust alignment,Hard,150
Aerospace,Turbine Machine,Vibration,Imbalance;Bearing wear,Check vibration,Balance;Replace bearing,Expert,240
Aerospace,Turbine Machine,Noise,Loose parts;Wear,Inspect parts,Tighten;Replace parts,Hard,180
General,Motor,Not starting,Power loss;Winding damage,Check voltage;Check winding,Restore power;Rewind motor,Medium,120
General,Motor,Overheating,Overload;Cooling issue,Check load;Check fan,Reduce load;Fix cooling,Medium,90
General,Motor,Noise,Bearing wear;Misalignment,Check bearing,Replace bearing,Medium,120
General,Pump,No flow,Blockage;Air lock,Check pipe,Clear blockage,Easy,45
General,Pump,Low pressure,Leak;Impeller wear,Check pressure,Fix leak;Replace impeller,Medium,90
General,Pump,Overheating,Dry run;Friction,Check water,Fix supply,Medium,60
General,Generator,No output,AVR fault;Winding issue,Check AVR,Replace AVR,Hard,180
General,Generator,Voltage fluctuation,Load variation;AVR issue,Measure voltage,Stabilize load,Medium,90
General,Transformer,Overheating,Overload;Cooling failure,Check load,Reduce load,Medium,120
General,Transformer,Noise,Loose core;Vibration,Inspect core,Tighten core,Hard,180"""

def get_all_rules():
    reader = csv.DictReader(io.StringIO(CSV_DATA))
    return list(reader)

CATEGORY_MAP = {
    "CNC Manufacturing": "CNC",
    "Aerospace":         "Aerospace",
    "Aeronautical":      "Aerospace",
    "Energy Sector":     "Energy",
}

COMPONENT_MAP = {
    "CNC Lathe":          ("CNC", "Lathe"),
    "CNC Milling Machine":("CNC", "VMC"),
    "CNC Drilling Machine":("CNC", "VMC"),
    "Compressor":         ("Aerospace", "5-Axis CNC"),
    "Turbine":            ("Aerospace", "Turbine Machine"),
    "Fan":                ("Aerospace", "Turbine Machine"),
    "Combustion Chamber": ("Aerospace", "Turbine Machine"),
    "Gearbox":            ("Energy",    "VFD"),
    "Generator":          ("General",   "Generator"),
    "Blades":             ("General",   "Motor"),
    "Bearings":           ("General",   "Motor"),
}

def get_faults_for_component(industry, component):
    rules = get_all_rules()
    mapping = COMPONENT_MAP.get(component)
    if not mapping:
        cat = CATEGORY_MAP.get(industry, "General")
        return [r for r in rules if r["category"] == cat]
    cat, mtype = mapping
    return [r for r in rules if r["category"] == cat and r["machine_type"] == mtype]

def get_preventive_tips(industry, component):
    tips = {
        "CNC Lathe": [
            "Check spindle oil level daily",
            "Inspect cutting tool for wear every 4 hours",
            "Verify coolant flow before starting",
            "Clean chip conveyor at end of shift",
            "Check belt tension weekly",
        ],
        "CNC Milling Machine": [
            "Inspect tool holders for runout daily",
            "Check axis lubrication every shift",
            "Verify workpiece clamping before operation",
            "Clean coolant filters weekly",
            "Check backlash compensation monthly",
        ],
        "CNC Drilling Machine": [
            "Check drill bit sharpness before each job",
            "Verify spindle speed matches material spec",
            "Inspect coolant nozzle alignment daily",
            "Check feed rate settings before operation",
            "Clean drill press table after each shift",
        ],
        "Compressor": [
            "Monitor pressure ratio every 2 hours",
            "Check inlet guide vane position daily",
            "Inspect compressor seals weekly",
            "Monitor vibration signature trends",
            "Check bearing temperatures hourly",
        ],
        "Turbine": [
            "Monitor exhaust gas temperature continuously",
            "Check turbine blade clearance weekly",
            "Inspect combustion liner monthly",
            "Monitor vibration levels every flight cycle",
            "Check oil consumption trends daily",
        ],
        "Fan": [
            "Inspect fan blade leading edges for erosion",
            "Check fan blade tip clearance monthly",
            "Monitor N1 speed during startup",
            "Inspect spinner for cracks weekly",
            "Check fan case for foreign object damage",
        ],
        "Combustion Chamber": [
            "Monitor EGT spread between thermocouples",
            "Check fuel nozzle flow rate monthly",
            "Inspect combustion liner for hot spots",
            "Monitor pressure oscillation frequency",
            "Check igniter condition every 500 cycles",
        ],
        "Gearbox": [
            "Check oil level and quality daily",
            "Monitor gear meshing vibration signature",
            "Inspect oil filter every 500 hours",
            "Check for metallic particles in oil",
            "Monitor bearing temperature continuously",
        ],
        "Generator": [
            "Check winding insulation resistance monthly",
            "Monitor output voltage and frequency",
            "Inspect slip rings and brushes weekly",
            "Check cooling fan operation daily",
            "Monitor power factor continuously",
        ],
        "Blades": [
            "Inspect blade surface for erosion weekly",
            "Check pitch control system monthly",
            "Monitor structural vibration continuously",
            "Inspect leading edge for ice formation",
            "Check blade balance after maintenance",
        ],
        "Bearings": [
            "Monitor bearing temperature every hour",
            "Check lubrication oil viscosity weekly",
            "Inspect seal condition monthly",
            "Monitor vibration spectrum daily",
            "Check alignment after any maintenance",
        ],
    }
    return tips.get(component, [
        "Perform visual inspection daily",
        "Check lubrication levels weekly",
        "Monitor temperature trends continuously",
        "Schedule preventive maintenance monthly",
        "Keep maintenance logs updated",
    ])
