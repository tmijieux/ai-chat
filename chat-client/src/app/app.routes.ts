import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'chat', pathMatch: 'full' }, // Default route

  // This line implements the lazy loading
  {
    path: 'chat',
    loadComponent: () => import('../components/chat/chat.component').then((m) => m.ChatComponent),
  },
  // Add other routes (e.g., { path: 'settings', component: SettingsComponent })
];
