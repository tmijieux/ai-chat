import { Component, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AppStatusService } from '../services/app-status.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('chat-client');

  constructor() {
    inject(AppStatusService).startPolling()
  }
}
