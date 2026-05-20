import { Routes } from '@angular/router'

export const routes: Routes = [
  { path: '', redirectTo: 'chat', pathMatch: 'full' }, // Default route

  // This line implements the lazy loading
  {
    path: 'chat',
    loadComponent: () => import('../components/chat/chat.component').then((m) => m.ChatComponent),
  },
  {
    path: 'settings',
    loadComponent: () =>
      import('../components/settings/settings.component').then((m) => m.SettingsComponent),
  },
]
