import { Component, input } from '@angular/core'
import { DiffLine } from '../../types/message-types'

@Component({
  selector: 'app-diff-block',
  standalone: true,
  templateUrl: './diff-block.component.html',
  styleUrls: ['./diff-block.component.scss'],
})
export class DiffBlockComponent {
  readonly lines = input.required<DiffLine[]>()
}
