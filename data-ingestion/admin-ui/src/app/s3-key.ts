export const BUCKET = 'dats-dhanush-f2026';
export const REGION = 'us-east-1';

export const FAMILIES = [
  'ACS880-01',
  'ACS880-04',
  'ACS880-04XT',
  'ACS880-07',
  'ACS880-11',
  'ACS880-14',
  'ACS880-17',
  'ACS880-31',
  'ACS880-34',
  'ACS880-37',
  'ACS880-104',
  'ACS880-107',
  'firmware-AINLX',
  'firmware-YINLX',
];

export function batchStamp(now = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}`;
}

export function safePdfName(filename: string): string {
  const base = filename.split(/[/\\]/).pop() ?? 'manual.pdf';
  const cleaned = base.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^\.+/, '');
  return cleaned.toLowerCase().endsWith('.pdf') ? cleaned : `${cleaned}.pdf`;
}

export function rawPrefix(family: string, batchId: string): string {
  return `torque-diagnostics/raw/${family}/${batchId}/`;
}

export function objectKey(family: string, batchId: string, filename: string, taken: Set<string>): string {
  const prefix = rawPrefix(family, batchId);
  let name = safePdfName(filename);
  const dot = name.lastIndexOf('.');
  const stem = name.slice(0, dot);
  const ext = name.slice(dot);
  let n = 2;
  while (taken.has(prefix + name)) {
    name = `${stem}-${n}${ext}`;
    n += 1;
  }
  taken.add(prefix + name);
  return prefix + name;
}

export function isPdf(file: File): boolean {
  const name = file.name.toLowerCase();
  return name.endsWith('.pdf') || file.type === 'application/pdf';
}
