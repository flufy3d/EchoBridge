# Show how much kernel pool the NVIDIA driver (pool tag NVRM) holds.
#
# Every Linux nvidia-smi / NVML session opened inside WSL2 leaks 21 NVRM
# objects (~7.5 KiB) in the Windows driver until Windows reboots; see
# docs/WSL_GPU_LEAK.md.  A freshly booted host holds a few MiB; after 41 days
# of uptime this host held 1,958 MiB in 7.27 million objects.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\nvrm-pool.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File deploy\windows\nvrm-pool.ps1 -Top 10
#
# Read-only; no administrator rights needed.  -Top N also lists the N largest
# non-paged pool tags (what poolmon shows).
param([int]$Top = 0)

$ErrorActionPreference = "Stop"
Add-Type -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class EchoBridgePoolTags {
    [DllImport("ntdll.dll")] static extern int NtQuerySystemInformation(int cls, IntPtr buf, int len, out int ret);
    public class Tag { public string Name; public long NonPaged; public long Objects; }
    // SystemPoolTagInformation (0x16): ULONG count, then 40-byte SYSTEM_POOLTAG
    // entries on x64 (tag, paged allocs/frees/used, non-paged allocs/frees/used).
    public static List<Tag> Read() {
        int len = 1 << 20, ret;
        while (true) {
            IntPtr buf = Marshal.AllocHGlobal(len);
            try {
                int st = NtQuerySystemInformation(0x16, buf, len, out ret);
                if (st == unchecked((int)0xC0000004)) { len = Math.Max(len * 2, ret + 4096); continue; }
                if (st != 0) throw new InvalidOperationException("NtQuerySystemInformation failed: 0x" + st.ToString("X8"));
                List<Tag> tags = new List<Tag>();
                int count = Marshal.ReadInt32(buf);
                for (int i = 0; i < count; i++) {
                    IntPtr e = IntPtr.Add(buf, 8 + i * 40);
                    byte[] name = new byte[4];
                    Marshal.Copy(e, name, 0, 4);
                    Tag t = new Tag();
                    t.Name = System.Text.Encoding.ASCII.GetString(name);
                    t.Objects = (long)(uint)Marshal.ReadInt32(e, 24) - (uint)Marshal.ReadInt32(e, 28);
                    t.NonPaged = Marshal.ReadInt64(e, 32);
                    tags.Add(t);
                }
                return tags;
            } finally { Marshal.FreeHGlobal(buf); }
        }
    }
}
"@

$tags = [EchoBridgePoolTags]::Read()
$nvrm = $tags | Where-Object { $_.Name -eq "NVRM" } | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$uptime = (Get-Date) - $os.LastBootUpTime
if ($nvrm) {
    "NVRM non-paged pool: {0:N0} MiB in {1:N0} objects (Windows uptime {2:N1} days)" -f ($nvrm.NonPaged / 1MB), $nvrm.Objects, $uptime.TotalDays
} else {
    "NVRM pool tag not found (no NVIDIA driver loaded?)"
}
if ($Top -gt 0) {
    $tags | Sort-Object NonPaged -Descending | Select-Object -First $Top |
        ForEach-Object { "  {0}  {1,8:N1} MiB  {2,12:N0} objects" -f $_.Name, ($_.NonPaged / 1MB), $_.Objects }
}
