#!/bin/bash

set -euo pipefail

TARGET_DIR="${1:-.}"
MAX_BYTES=$((2 * 1024 * 1024))

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! have_cmd sips; then
  echo "Error: sips is required and should already exist on macOS." >&2
  exit 1
fi

if ! have_cmd magick; then
  echo "Error: ImageMagick is required. Install with: brew install imagemagick" >&2
  exit 1
fi

if ! have_cmd pngquant; then
  echo "Warning: pngquant not found. PNG compression will be less effective." >&2
fi

get_dimensions() {
  local file="$1"
  sips -g pixelWidth -g pixelHeight "$file" 2>/dev/null | awk '
    /pixelWidth:/ {w=$2}
    /pixelHeight:/ {h=$2}
    END {print w, h}
  '
}

resize_with_sips() {
  local input="$1"
  local output="$2"
  local max_w="$3"
  local max_h="$4"

  cp "$input" "$output"
  sips -Z "$max_w" "$output" >/dev/null 2>&1

  read -r new_w new_h < <(get_dimensions "$output")
  if [[ -n "${new_h:-}" && "$new_h" -gt "$max_h" ]]; then
    sips -Z "$max_h" "$output" >/dev/null 2>&1
  fi
}

compress_jpeg_like() {
  local input="$1"
  local output="$2"

  local tmpdir
  tmpdir="$(mktemp -d)"

  local working="$tmpdir/working.jpg"
  local candidate="$tmpdir/candidate.jpg"

  magick "$input" -auto-orient "$working"

  read -r orig_w orig_h < <(get_dimensions "$working")
  local scale=100
  local quality=85

  while :; do
    local target_w=$(( orig_w * scale / 100 ))
    local target_h=$(( orig_h * scale / 100 ))
    (( target_w < 1 )) && target_w=1
    (( target_h < 1 )) && target_h=1

    resize_with_sips "$working" "$candidate" "$target_w" "$target_h"
    magick "$candidate" -strip -interlace Plane -quality "$quality" "$candidate"

    local size
    size=$(stat -f%z "$candidate")

    if (( size <= MAX_BYTES )); then
      mv "$candidate" "$output"
      rm -rf "$tmpdir"
      return 0
    fi

    if (( quality > 40 )); then
      quality=$(( quality - 10 ))
    elif (( scale > 35 )); then
      scale=$(( scale - 10 ))
      quality=85
    else
      mv "$candidate" "$output"
      rm -rf "$tmpdir"
      return 0
    fi
  done
}

compress_png() {
  local input="$1"
  local output="$2"

  local tmpdir
  tmpdir="$(mktemp -d)"

  local working="$tmpdir/working.png"
  local resized="$tmpdir/resized.png"
  local candidate="$tmpdir/candidate.png"

  magick "$input" -auto-orient "$working"

  read -r orig_w orig_h < <(get_dimensions "$working")
  local scale=100

  while :; do
    local target_w=$(( orig_w * scale / 100 ))
    local target_h=$(( orig_h * scale / 100 ))
    (( target_w < 1 )) && target_w=1
    (( target_h < 1 )) && target_h=1

    resize_with_sips "$working" "$resized" "$target_w" "$target_h"

    if have_cmd pngquant; then
      pngquant --force --skip-if-larger --output "$candidate" 256 "$resized" >/dev/null 2>&1 || cp "$resized" "$candidate"
    else
      magick "$resized" -strip -define png:compression-level=9 "$candidate"
    fi

    local size
    size=$(stat -f%z "$candidate")

    if (( size <= MAX_BYTES )); then
      mv "$candidate" "$output"
      rm -rf "$tmpdir"
      return 0
    fi

    if (( scale > 35 )); then
      scale=$(( scale - 10 ))
    else
      mv "$candidate" "$output"
      rm -rf "$tmpdir"
      return 0
    fi
  done
}

compress_webp() {
  local input="$1"
  local output="$2"

  local tmpdir
  tmpdir="$(mktemp -d)"

  local working="$tmpdir/working.webp"
  local candidate="$tmpdir/candidate.webp"

  magick "$input" -auto-orient "$working"

  read -r orig_w orig_h < <(get_dimensions "$working")
  local scale=100
  local quality=80

  while :; do
    local target_w=$(( orig_w * scale / 100 ))
    local target_h=$(( orig_h * scale / 100 ))
    (( target_w < 1 )) && target_w=1
    (( target_h < 1 )) && target_h=1

    resize_with_sips "$working" "$candidate" "$target_w" "$target_h"
    magick "$candidate" -strip -quality "$quality" "$candidate"

    local size
    size=$(stat -f%z "$candidate")

    if (( size <= MAX_BYTES )); then
      mv "$candidate" "$output"
      rm -rf "$tmpdir"
      return 0
    fi

    if (( quality > 35 )); then
      quality=$(( quality - 10 ))
    elif (( scale > 35 )); then
      scale=$(( scale - 10 ))
      quality=80
    else
      mv "$candidate" "$output"
      rm -rf "$tmpdir"
      return 0
    fi
  done
}

process_file() {
  local file="$1"
  local ext="${file##*.}"
  local dir
  dir="$(dirname "$file")"
  local base
  base="$(basename "$file")"

  local tmp_out
  tmp_out="$(mktemp "${dir}/.${base}.tmp.XXXXXX")"
  local backup="${file}.bak"

  cp "$file" "$backup"

  case "$ext" in
    jpg|JPG|jpeg|JPEG)
      compress_jpeg_like "$file" "$tmp_out"
      ;;
    png|PNG)
      compress_png "$file" "$tmp_out"
      ;;
    webp|WEBP)
      compress_webp "$file" "$tmp_out"
      ;;
    *)
      rm -f "$tmp_out"
      rm -f "$backup"
      return 0
      ;;
  esac

  local final_size
  final_size=$(stat -f%z "$tmp_out")
  local original_size
  original_size=$(stat -f%z "$file")

  # Only replace if smaller
  if (( final_size < original_size )); then
    mv "$tmp_out" "$file"
    rm -f "$backup"
    if (( final_size > MAX_BYTES )); then
      echo "Warning: $file is still above 2MB ($(awk "BEGIN {printf \"%.2f\", $final_size/1024/1024}") MB)"
    else
      echo "Compressed: $file -> $(awk "BEGIN {printf \"%.2f\", $final_size/1024/1024}") MB"
    fi
  else
    rm -f "$tmp_out"
    mv "$backup" "$file"
    echo "Skipped: $file (compression was not smaller)"
  fi
}

export -f have_cmd get_dimensions resize_with_sips
export -f compress_jpeg_like compress_png compress_webp process_file
export MAX_BYTES

find "$TARGET_DIR" -type f \( \
  -iname "*.jpg" -o -iname "*.jpeg" -o \
  -iname "*.png" -o \
  -iname "*.webp" \
\) ! -name "*.bak" | while IFS= read -r file; do
  process_file "$file"
done

echo "Done."
