#!/bin/bash
# last usage: 25.09.2025

# Script: 01_reprocess_win.sh
#
# Description:
# This script processes radiosonde data from the ORCESTRA campaign (Winkler set).
# It performs:
#   1. Level 0 → Level 1 conversion of raw files (.mwx/.cor) to standardized NetCDF
#   2. Level 1 → Level 2 processing (quality control, formatting)
#   3. Concatenation of Level 2 files into a single dataset with updated metadata
#
# Usage:
#   bash 01_reprocess_win.sh                 # Run full processing chain (default)
#   bash 01_reprocess_win.sh --only-l0-l1    # Run only Level 0 to Level 1 processing
#   bash 01_reprocess_win.sh --only-l1-l2    # Run only Level 1 to Level 2 processing
#   bash 01_reprocess_win.sh --only-post-l2  # Run only concatenation and metadata update
#
# Dependencies:
#   - pysonde package (install via: pip install pysonde)
#   - nco tools (e.g., ncrcat, ncatted)
#   - sounding_converter (installed with pysonde)
#

# Install pysonde if not already installed
# pip install pysonde

#v4.0.4_raw --> Santander version, final fixed cor files
#v4.0.5_raw --> with linear_masked interpolation

DATASET_VERSION="RS_ORCESTRA_level2_v4.0.5_raw.nc"

BASE_PATH="/Users/marius/ownCloud/PhD/12_Orcestra_Campaign"
CONFIG_PATH="$BASE_PATH/pysonde_for_datapaper/pysonde/config"

DIRECTORY_LEVEL2="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level2"
NETCDF_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level2/merged_dataset/$DATASET_VERSION"
TEMP_FILE="temp.nc"

ls "$CONFIG_PATH"

# Default values: process both level0->level1, level1->level2, and post-processing
PROCESS_L0_L1=true
PROCESS_L1_L2=true
PROCESS_POST_L2=true

# Check command-line arguments
for arg in "$@"; do
    case $arg in
        --only-l0-l1)
            PROCESS_L0_L1=true
            PROCESS_L1_L2=false
            PROCESS_POST_L2=false
            ;;
        --only-l1-l2)
            PROCESS_L0_L1=false
            PROCESS_L1_L2=true
            PROCESS_POST_L2=false
            ;;
        --only-post-l2)
            PROCESS_L0_L1=false
            PROCESS_L1_L2=false
            PROCESS_POST_L2=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--only-l0-l1] [--only-l1-l2] [--only-post-l2]"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------------------------------------
# LEVEL 0 → LEVEL 1 PROCESSING
# -----------------------------------------------------------------------------------------------------------

if $PROCESS_L0_L1; then
    echo -e "\nStarting with level0 to level1 processing now."

    # Log files
    BCO_LOG_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files/process_01_BCO_L0-L1_log.txt"
    METEOR_LOG_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files/process_02_Meteor_L0-L1_log.txt"
    INMG_LOG_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files/process_03_INMG_L0-L1_log.txt"

    # ensure log dir exists
    mkdir -p "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files"
    # truncate logs for this run
    : > "$BCO_LOG_FILE"
    : > "$METEOR_LOG_FILE"
    : > "$INMG_LOG_FILE"

    # Processing BCO files
    BCO_FILES=("$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level0/BCO/"*.mwx)
    BCO_TOTAL=${#BCO_FILES[@]}

    echo -e "\nProcessing $BCO_TOTAL BCO files..." | tee -a "$BCO_LOG_FILE"

    BCO_COUNT=0
    for file in "${BCO_FILES[@]}"; do
        ((BCO_COUNT++))
        echo -e "\nProcessing file ($BCO_COUNT/$BCO_TOTAL): $file" | tee -a "$BCO_LOG_FILE"
        
        sounding_converter -i "$file" \
            -o "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level1/BCO/RS_{campaign}_{platform}_L1_%Y%m%dT%H%M_{direction}.nc" \
            -c "$CONFIG_PATH/main_BCO.yaml" \
            2>&1 | tee -a "$BCO_LOG_FILE"

        echo -e "\n--------------------------------------" | tee -a "$BCO_LOG_FILE"
    done
    echo -e "\nAll BCO files processed." | tee -a "$BCO_LOG_FILE"

    # Processing Meteor files
    METEOR_FILES=("$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level0/Meteor/"*.mwx)
    METEOR_TOTAL=${#METEOR_FILES[@]}

    echo -e "\nProcessing $METEOR_TOTAL Meteor files..." | tee -a "$METEOR_LOG_FILE"

    METEOR_COUNT=0
    for file in "${METEOR_FILES[@]}"; do
        ((METEOR_COUNT++))
        echo -e "\nProcessing file ($METEOR_COUNT/$METEOR_TOTAL): $file" | tee -a "$METEOR_LOG_FILE"
        
        sounding_converter -i "$file" \
            -o "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level1/Meteor/RS_{campaign}_{platform}_L1_%Y%m%dT%H%M_{direction}.nc" \
            -c "$CONFIG_PATH/main_Met.yaml" \
            2>&1 | tee -a "$METEOR_LOG_FILE"

        echo -e "\n--------------------------------------" | tee -a "$METEOR_LOG_FILE"
    done
    echo -e "\nAll Meteor files processed." | tee -a "$METEOR_LOG_FILE"

    # Processing INMG files
    INMG_FILES=("$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level0/INMG/"*.cor)
    INMG_TOTAL=${#INMG_FILES[@]}

    echo -e "\nProcessing $INMG_TOTAL INMG files..." | tee -a "$INMG_LOG_FILE"

    INMG_COUNT=0
    for file in "${INMG_FILES[@]}"; do
        ((INMG_COUNT++))
        echo -e "\nProcessing file ($INMG_COUNT/$INMG_TOTAL): $file" | tee -a "$INMG_LOG_FILE"
        
        sounding_converter -i "$file" \
            -o "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level1/INMG/RS_{campaign}_{platform}_L1_%Y%m%dT%H%M_{direction}.nc" \
            -c "$CONFIG_PATH/main_INMG.yaml" \
            2>&1 | tee -a "$INMG_LOG_FILE"

        echo -e "\n--------------------------------------" | tee -a "$INMG_LOG_FILE"
    done
    echo -e "\nAll INMG files processed." | tee -a "$INMG_LOG_FILE"
fi

# -----------------------------------------------------------------------------------------------------------
# LEVEL 1 → LEVEL 2 PROCESSING
# -----------------------------------------------------------------------------------------------------------

if $PROCESS_L1_L2; then
    echo -e "\nStarting with level1 to level2 processing now."

    # Log files
    BCO_L2_LOG_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files/process_04_BCO_L1-L2_log.txt"
    METEOR_L2_LOG_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files/process_05_Meteor_L1-L2_log.txt"
    INMG_L2_LOG_FILE="$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files/process_06_INMG_L1-L2_log.txt"
    
    # ensure log dir exists
    mkdir -p "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/reprocessing_log_files"
    # truncate logs for this run
    : > "$BCO_L2_LOG_FILE"
    : > "$METEOR_L2_LOG_FILE"
    : > "$INMG_L2_LOG_FILE"

    # Processing BCO files
    BCO_L2_FILES=("$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level1/BCO/"*.nc)
    BCO_L2_TOTAL=${#BCO_L2_FILES[@]}

    echo -e "\nProcessing $BCO_L2_TOTAL BCO level1 to level2 files..." | tee -a "$BCO_L2_LOG_FILE"

    BCO_L2_COUNT=0
    for file in "${BCO_L2_FILES[@]}"; do
        ((BCO_L2_COUNT++))
        echo -e "\nProcessing file ($BCO_L2_COUNT/$BCO_L2_TOTAL): $file" | tee -a "$BCO_L2_LOG_FILE"
        
        sounding_converter -i "$file" \
            -o "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level2/RS_{campaign}_{platform}_L2_%Y%m%dT%H%M_{direction}.nc" \
            -c "$CONFIG_PATH/main_BCO.yaml" \
            2>&1 | tee -a "$BCO_L2_LOG_FILE"

        echo -e "\n--------------------------------------" | tee -a "$BCO_L2_LOG_FILE"
    done
    echo -e "\nAll BCO level1 to level2 files processed." | tee -a "$BCO_L2_LOG_FILE"
    
    # Repeat for Meteor files
    METEOR_L2_FILES=("$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level1/Meteor/"*.nc)
    METEOR_L2_TOTAL=${#METEOR_L2_FILES[@]}

    echo -e "\nProcessing $METEOR_L2_TOTAL Meteor level1 to level2 files..." | tee -a "$METEOR_L2_LOG_FILE"

    METEOR_L2_COUNT=0
    for file in "${METEOR_L2_FILES[@]}"; do
        ((METEOR_L2_COUNT++))
        echo -e "\nProcessing file ($METEOR_L2_COUNT/$METEOR_L2_TOTAL): $file" | tee -a "$METEOR_L2_LOG_FILE"
        
        sounding_converter -i "$file" \
            -o "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level2/RS_{campaign}_RV_Meteor_L2_%Y%m%dT%H%M_{direction}.nc" \
            -c "$CONFIG_PATH/main_Met.yaml" \
            2>&1 | tee -a "$METEOR_L2_LOG_FILE"

        echo -e "\n--------------------------------------" | tee -a "$METEOR_L2_LOG_FILE"
    done
    echo -e "\nAll Meteor level1 to level2 files processed." | tee -a "$METEOR_L2_LOG_FILE"

    # Processing INMG files
    INMG_L2_FILES=("$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level1/INMG/"*.nc)
    INMG_L2_TOTAL=${#INMG_L2_FILES[@]}

    echo -e "\nProcessing $INMG_L2_TOTAL INMG level1 to level2 files..." | tee -a "$INMG_L2_LOG_FILE"

    INMG_L2_COUNT=0
    for file in "${INMG_L2_FILES[@]}"; do
        ((INMG_L2_COUNT++))
        echo -e "\nProcessing file ($INMG_L2_COUNT/$INMG_L2_TOTAL): $file" | tee -a "$INMG_L2_LOG_FILE"
        
        sounding_converter -i "$file" \
            -o "$BASE_PATH/00_ORCESTRA_Radiosondes_Winkler/level2/RS_{campaign}_{platform}_L2_%Y%m%dT%H%M_{direction}.nc" \
            -c "$CONFIG_PATH/main_INMG.yaml" \
            2>&1 | tee -a "$INMG_L2_LOG_FILE"

        echo -e "\n--------------------------------------" | tee -a "$INMG_L2_LOG_FILE"
    done
    echo -e "\nAll INMG level1 to level2 files processed." | tee -a "$INMG_L2_LOG_FILE"
fi

# -----------------------------------------------------------------------------------------------------------
# LEVEL 2 → CONCATENATED DATASET
# -----------------------------------------------------------------------------------------------------------

if $PROCESS_POST_L2; then
    echo -e "\n🔹 Starting post-processing of Level 2 data..."

    ## Step 1: Add Launch Platform Coordinate
    # echo -e "\n🚀 Adding launch platform as a coordinate..."
    # python "$BASE_PATH1/add_platform_coordinate.py" "$DIRECTORY_LEVEL2"

    ## Step 2: Concatenate Level 2 NetCDF files
    echo -e "\n📂 Gathering Level 2 NetCDF files for concatenation..."
    files=$(find "$DIRECTORY_LEVEL2" -name "RS_*L2_*.nc")

    if [ -z "$files" ]; then
        echo "⚠️ No files found matching RS_*L2_*.nc in $DIRECTORY_LEVEL2. Exiting."
        exit 1
    fi

    echo -e "\n🔗 Concatenating all L2 files into $NETCDF_FILE..."
    ncrcat -h $files "$NETCDF_FILE"

    ## Step 3: Modify Attributes in the NetCDF File
    echo -e "\n🔧 Adapting attributes after concatenation..."
    
    # Extract current history attribute
    current_history=$(ncdump -h "$NETCDF_FILE" | grep ":history =" | cut -d '"' -f2)

    # Ensure "Meteomodem Eoscan;" is added only if not present
    if [[ "$current_history" != *"Meteomodem Eoscan"* ]]; then
        new_history="Meteomodem Eoscan Software (2.1.241218); $current_history"
    else
        new_history="$current_history"
    fi

    # Create a clean copy of the NetCDF file
    cp "$NETCDF_FILE" "$TEMP_FILE"

    # Delete all existing attributes to reset the order #### -a featureType,global,d,, \
    ncatted -O -h \
        -a title,global,d,, \
        -a summary,global,d,, \
        -a creator_name,global,d,, \
        -a creator_email,global,d,, \
        -a project,global,d,, \
        -a platform,global,d,, \
        -a source,global,d,, \
        -a history,global,d,, \
        -a license,global,d,, \
        -a references,global,d,, \
        -a keywords,global,d,, \
        "$TEMP_FILE"

    # Re-add attributes in the correct order #### #-a featureType,globale,c,c,"profile" \
    ncatted -O -h \
        -a title,global,c,c,"RAPSODI Radiosonde Measurements during ORCESTRA (Level 2)" \
        -a summary,global,c,c,"Vertical atmospheric profile, retrieved from atmospheric sounding attached to a helium-filled balloon." \
        -a creator_name,global,c,c,"Marius Winkler, Marius Rixen" \
        -a creator_email,global,c,c,"marius.winkler@mpimet.mpg.de, marius.rixen@mpimet.mpg.de" \
        -a project,global,c,c,"ORCESTRA, PERCUSION, BOW-TIE, PICCOLO, SCORE, MAESTRO" \
        -a platform,global,c,c,"INMG, RV Meteor, BCO" \
        -a source,global,c,c,"Radiosondes" \
        -a history,global,c,c,"$new_history" \
        -a license,global,c,c,"CC-BY-4.0" \
        -a references,global,c,c,"https://github.com/observingClouds/pysonde" \
        -a keywords,global,c,c,"ORCESTRA, RAPSODI, Radiosonde Profiles, Sounding, INMG, RV Meteor, BCO" \
        "$TEMP_FILE"

    # Overwrite the original file with the corrected version
    mv "$TEMP_FILE" "$NETCDF_FILE"

    echo -e "\n✅ Post-processing completed successfully."
else
    echo -e "\n⏭ Skipping post-processing of Level 2 data as per user setting."
fi
