#!/bin/bash
DATASET_NAME="DIV2K_HR"
SOURCE_DIR="/home/xi9/code/Bezier_splatting/output"
SAVE_DIR="/project/siyuh/common/xiliu/Outputs"

echo $SOURCE_DIR
FAST_DIFFVG_DIRS=$(find -L "$SOURCE_DIR" -maxdepth 1 -type d -name "bezier_splatting_area_our_*")
echo $FAST_DIFFVG_DIRS
for LINE_DIR in $FAST_DIFFVG_DIRS; do
    LINE_NAME=$(basename "$LINE_DIR")
    echo $LINE_NAME
    for SUB_DIR in "$LINE_DIR/$DATASET_NAME"/*; do
        if [ -d "$SUB_DIR" ]; then 
            DIR_NAME=$(basename "$SUB_DIR") 
            FORMATTED_DIR=$DIR_NAME
            SRC_FILE="$SUB_DIR/final.png"
            DEST_PATH="${SAVE_DIR}/${LINE_NAME}/${DATASET_NAME}/${FORMATTED_DIR}.png"
            if [ -f "$SRC_FILE" ]; then
                mkdir -p "$(dirname "$DEST_PATH")"
                cp "$SRC_FILE" "$DEST_PATH"
                echo "Copied: $SRC_FILE -> $DEST_PATH"
            fi
        fi
    done
done
