#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
cd kernel/dtb

DTS=tegra210-p3448-0002-p3449-0000-b00.dts
DTB=tegra210-p3448-0002-p3449-0000-b00.dtb

echo "[INFO] Decompiling $DTB to $DTS..."
sudo dtc -I dtb -O dts -o "$DTS" "$DTB"

fix_spi_node() {
local node=$1
if grep -A5 "$node {" "$DTS" | grep -q 'status'; then
sudo sed -i \
'/'"$node"' {/,/spi-max-frequency/ s/status = "[^"]*";/status = "okay";/' \
"$DTS"
else
sudo sed -i \
'/'"$node"' {/,/spi-max-frequency/ s/compatible = "tegra-spidev"/&\
status = "okay";/' \
"$DTS"
fi
}

echo "[ACTION] Enabling spi@0 and spi@1..."
fix_spi_node "spi@0"
fix_spi_node "spi@1"

patch_pin() {
local pin=$1
sudo sed -i \
'/'"$pin"' {/,/nvidia,enable-input/ {
s/nvidia,function = "rsvd1"/nvidia,function = "spi1"/
s/nvidia,tristate = <0x01>/nvidia,tristate = <0x00>/
s/nvidia,enable-input = <0x00>/nvidia,enable-input = <0x01>/
}' \
"$DTS"
}

echo "[ACTION] Patching pinmux blocks..."
for pin in spi1_mosi_pc0 spi1_miso_pc1 spi1_sck_pc2 spi1_cs0_pc3 spi1_cs1_pc4; do
patch_pin "$pin"
done

echo "[ACTION] Fixing tristate and input-enable for SPI1 pins..."

fix_pinmux_field() {
local pin=$1
awk -v pin="$pin" '
BEGIN { in_block = 0 }
{
if ($0 ~ pin " {") { in_block = 1 }
if (in_block && /nvidia,tristate =/)       { sub(/<0x1>/, "<0x0>") }
if (in_block && /nvidia,enable-input =/)   { sub(/<0x0>/, "<0x1>") }
print
if (in_block && /}/) { in_block = 0 }
}' "$DTS" | sudo tee "$DTS.fixed" > /dev/null && sudo mv "$DTS.fixed" "$DTS"
}

fix_pinmux_field "spi1_mosi_pc0"
fix_pinmux_field "spi1_miso_pc1"
fix_pinmux_field "spi1_sck_pc2"
fix_pinmux_field "spi1_cs0_pc3"
fix_pinmux_field "spi1_cs1_pc4"

echo "[INFO] Recompiling DTS to $DTB..."
sudo dtc -I dts -O dtb -o "$DTB" "$DTS"

echo "[OK] SPI DTS patch applied and DTB regenerated successfully."