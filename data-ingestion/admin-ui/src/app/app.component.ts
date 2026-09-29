import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import {
  BUCKET,
  FAMILIES,
  REGION,
  batchStamp,
  isPdf,
  objectKey,
  rawPrefix,
} from './s3-key';

interface Ticket {
  file: File;
  key: string;
  state: 'held' | 'sending' | 'seated' | 'refused';
  note: string;
}

interface SessionCreds {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken: string;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  readonly bucket = BUCKET;
  readonly region = REGION;
  readonly families = FAMILIES;

  family = FAMILIES[0];
  batchId = batchStamp();
  tickets: Ticket[] = [];
  dragOver = false;
  busy = false;
  banner = '';

  accessKeyId = '';
  secretAccessKey = '';
  sessionToken = '';
  armed = false;

  constructor() {
    const saved = sessionStorage.getItem('torque-bay-session');
    if (!saved) {
      return;
    }
    try {
      const creds = JSON.parse(saved) as SessionCreds;
      this.accessKeyId = creds.accessKeyId;
      this.secretAccessKey = creds.secretAccessKey;
      this.sessionToken = creds.sessionToken;
      this.armed = Boolean(creds.accessKeyId && creds.secretAccessKey);
    } catch {
      sessionStorage.removeItem('torque-bay-session');
    }
  }

  get prefix(): string {
    return rawPrefix(this.family, this.batchId);
  }

  get markerKey(): string {
    return `${this.prefix}_COMPLETE`;
  }

  arm(): void {
    if (!this.accessKeyId.trim() || !this.secretAccessKey.trim()) {
      this.banner = 'The bay stays dark until both key fields are filled.';
      this.armed = false;
      return;
    }
    const creds: SessionCreds = {
      accessKeyId: this.accessKeyId.trim(),
      secretAccessKey: this.secretAccessKey.trim(),
      sessionToken: this.sessionToken.trim(),
    };
    sessionStorage.setItem('torque-bay-session', JSON.stringify(creds));
    this.armed = true;
    this.banner = 'Session is armed for this browser tab only.';
  }

  disarm(): void {
    sessionStorage.removeItem('torque-bay-session');
    this.accessKeyId = '';
    this.secretAccessKey = '';
    this.sessionToken = '';
    this.armed = false;
    this.banner = 'Session cleared from this tab.';
  }

  freshBatch(): void {
    this.batchId = batchStamp();
    this.tickets = [];
    this.banner = `New sleeve ${this.batchId}.`;
  }

  onFamilyChange(): void {
    this.rekey();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver = false;
    this.takeFiles(event.dataTransfer?.files);
  }

  onPick(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.takeFiles(input.files);
    input.value = '';
  }

  async send(): Promise<void> {
    if (!this.armed || this.tickets.length === 0 || this.busy) {
      return;
    }
    this.busy = true;
    this.banner = '';
    const client = new S3Client({
      region: REGION,
      credentials: {
        accessKeyId: this.accessKeyId.trim(),
        secretAccessKey: this.secretAccessKey.trim(),
        sessionToken: this.sessionToken.trim() || undefined,
      },
    });
    try {
      for (const ticket of this.tickets) {
        ticket.state = 'sending';
        try {
          await client.send(new PutObjectCommand({
            Bucket: BUCKET,
            Key: ticket.key,
            Body: new Uint8Array(await ticket.file.arrayBuffer()),
            ContentType: 'application/pdf',
          }));
          ticket.state = 'seated';
          ticket.note = 'in the raw sleeve';
        } catch (err) {
          ticket.state = 'refused';
          ticket.note = err instanceof Error ? err.message : 'upload refused';
        }
      }
      const seated = this.tickets.filter((ticket) => ticket.state === 'seated');
      if (seated.length === this.tickets.length) {
        await client.send(new PutObjectCommand({
          Bucket: BUCKET,
          Key: this.markerKey,
          Body: seated.map((ticket) => ticket.key).join('\n'),
          ContentType: 'text/plain',
        }));
        this.banner = `Sleeve closed. ${seated.length} manuals seated, marker written.`;
      } else {
        this.banner = 'Sleeve left open. The marker is written only when every file seats.';
      }
    } finally {
      this.busy = false;
    }
  }

  private takeFiles(list: FileList | null | undefined): void {
    if (!list) {
      return;
    }
    const taken = new Set(this.tickets.map((ticket) => ticket.key));
    let skipped = 0;
    for (const file of Array.from(list)) {
      if (!isPdf(file)) {
        skipped += 1;
        continue;
      }
      this.tickets.push({
        file,
        key: objectKey(this.family, this.batchId, file.name, taken),
        state: 'held',
        note: 'waiting on the bench',
      });
    }
    this.banner = skipped
      ? `${skipped} file${skipped === 1 ? '' : 's'} refused. This bay takes PDFs.`
      : '';
  }

  private rekey(): void {
    const taken = new Set<string>();
    this.tickets = this.tickets.map((ticket) => ({
      ...ticket,
      key: objectKey(this.family, this.batchId, ticket.file.name, taken),
      state: 'held',
      note: 'key rewritten for the new family',
    }));
  }
}
